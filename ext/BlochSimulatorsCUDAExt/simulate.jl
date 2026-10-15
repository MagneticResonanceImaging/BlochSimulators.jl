# GPU methods of the simulation drivers in src/simulate/ and
# src/derivatives/forward_sensitivity.jl: output allocation and the CUDA
# kernels. EPG simulations assign one warp (WARPSIZE threads) to each voxel,
# isochromat simulations one thread (see below); the signal kernel assigns one
# thread to each time point.

"""
However, if the `parameters` are a CuArray, the simulation is performed on the GPU.
"""
function simulate_magnetization(sequence, parameters::CuArray)
    simulate_magnetization(CUDALibs(), gpu(sequence), parameters)
end

function simulate_magnetization!(magnetization, ::CUDALibs, sequence, parameters)
    # GPU execution: Launch CUDA kernel with multiple threads per voxel
    # Each voxel gets assigned `WARPSIZE` threads, over which the EPG states are spread
    # Since `THREADS_PER_BLOCK` is fixed, the number of required blocks is calculated
    nr_voxels = length(parameters)
    nr_blocks = cld(nr_voxels * WARPSIZE, THREADS_PER_BLOCK)

    # Define kernel function (executed on GPU)
    magnetization_kernel!(magnetization, sequence, parameters) = begin

        # Calculate voxel index from block and thread indices
        voxel = cld((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x, WARPSIZE)

        # Initialize state that gets updated during time integration
        states = initialize_states(CUDALibs(), sequence)

        # Do nothing if voxel index is out of bounds (extra threads in last block)
        if voxel > length(parameters)
            return nothing
        end

        # Run simulation for this voxel
        simulate_magnetization!(
            view(magnetization, :, voxel),
            sequence,
            states,
            @inbounds parameters[voxel]
        )
    end

    # Launch kernel and synchronize
    CUDA.@sync begin
        @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK magnetization_kernel!(magnetization, sequence, parameters)
    end
    return nothing
end

function _allocate_array_on_resource(::CUDALibs, _eltype, _size)
    return CUDA.zeros(_eltype, _size...)
end

function _signal_per_coil!(signal, ::CUDALibs, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    # GPU execution: Launch CUDA kernel with one thread per time point
    nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

    # Define kernel function to be run by each thread on GPU
    magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coordinates, coil_sensitivities) = begin

        # Calculate time point index from block and thread indices
        time_point = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        # Compute signal at this time point if within bounds
        if time_point <= nsamples(trajectory)
            signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
        end
        return nothing
    end

    # Launch kernel and synchronize
    CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
end

function _simulate!(magnetization, ∂magnetization::NamedTuple{derivatives}, ::CUDALibs, sequence, parameters) where {derivatives}

    function kernel!(magnetization, ∂magnetization, sequence, parameters)
        voxel = cld((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x, WARPSIZE)
        Ω = initialize_derivative_states(CUDALibs(), sequence, Val(derivatives))

        # extra threads in the last block have no voxel to simulate
        voxel > length(parameters) && return nothing

        output = (view(magnetization, :, voxel),
            tangents_of(map(∂m -> view(∂m, :, voxel), ∂magnetization)))
        simulate_magnetization!(output, sequence, Ω, @inbounds parameters[voxel])
        return nothing
    end

    nr_blocks = cld(length(parameters) * WARPSIZE, THREADS_PER_BLOCK)
    CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK kernel!(
        magnetization, ∂magnetization, sequence, parameters)
    return nothing
end


# Isochromat simulations
# =====================================================================
# An isochromat simulation has a single, small state, so one thread simulates
# one voxel. (With a warp per voxel, as for EPG, all 32 lanes would compute
# the same thing: measured 4x slower for pSSFP3D.)
#
# Neighbouring threads then simulate neighbouring voxels, and each writes the
# output of its voxel whenever it samples. In the (time, voxel) layout of the
# output, those writes are scattered: the 32 threads of a warp write to 32
# different places in memory, for every sample (and read too, since sampling
# adds to the output). The kernels therefore write into a (voxel, time)
# buffer, where the threads of a warp write next to each other, and the buffer
# is transposed into the output afterwards. Measured for pSSFP3D with 100k
# voxels and 1000 TRs, transposes included, this took the simulation from 30
# to 10 ms, and with T₁ and T₂ derivatives, which write three outputs, from
# 109 to 25 ms.
#
# The buffers are allocated per chunk of voxels, so that they do not double the
# memory that the output takes.

const VOXELS_PER_CHUNK = 2^16

"""
    _in_voxel_chunks(launch, outputs::Tuple)

For each chunk of voxels, call `launch(buffers, offset)` with a zeroed
(voxel, time...) buffer for each of the (time..., voxel) `outputs`, and the
index offset of the chunk's first voxel. Then transpose the buffers into the
chunk's part of the outputs.
"""
function _in_voxel_chunks(launch, outputs::Tuple)
    n = ndims(first(outputs))
    for voxels in Iterators.partition(1:size(first(outputs), n), VOXELS_PER_CHUNK)
        buffers = map(o -> CUDA.zeros(eltype(o), length(voxels), size(o)[1:n-1]...), outputs)
        launch(buffers, first(voxels) - 1)
        foreach(outputs, buffers) do output, buffer
            _transpose!(selectdim(output, n, voxels), buffer)
            CUDA.unsafe_free!(buffer)
        end
    end
    return nothing
end

"""
    _transpose!(output, buffer)

Copy a (voxel, time...) `buffer` into a (time..., voxel) `output`. For the
usual matrices, a tiled transpose through shared memory, so that both reads
and writes are coalesced: several times faster than `permutedims!` (measured
for 100k voxels and 1000 time points, allocating the buffer included: 21 → 5 ms).
"""
_transpose!(output, buffer) = permutedims!(output, buffer, (2:ndims(buffer)..., 1))

# A tile of 32x32 elements, copied by 32x8 threads. Constants, since the size
# of the shared memory must be known at compile time.
const TILE = 32
const ROWS = 8

function _transpose!(output::CuMatrix, buffer::CuMatrix)

    function kernel!(output, buffer)
        tile = CuStaticSharedArray(eltype(buffer), (TILE + 1, TILE)) # +1: no bank conflicts
        tx, ty = threadIdx().x, threadIdx().y
        voxel₀, time₀ = (blockIdx().x - 1) * TILE, (blockIdx().y - 1) * TILE
        for k in 0:ROWS:TILE-1
            voxel, time = voxel₀ + tx, time₀ + ty + k
            if voxel <= size(buffer, 1) && time <= size(buffer, 2)
                @inbounds tile[tx, ty+k] = buffer[voxel, time]
            end
        end
        sync_threads()
        for k in 0:ROWS:TILE-1
            time, voxel = time₀ + tx, voxel₀ + ty + k
            if time <= size(output, 1) && voxel <= size(output, 2)
                @inbounds output[time, voxel] = tile[ty+k, tx]
            end
        end
        return nothing
    end

    blocks = (cld(size(buffer, 1), TILE), cld(size(buffer, 2), TILE))
    CUDA.@sync @cuda blocks = blocks threads = (TILE, ROWS) kernel!(output, buffer)
    return output
end

# The part of a (voxel, time...) buffer that holds voxel `i`
@inline _voxel_output(buffer, i) = view(buffer, i, ntuple(_ -> Colon(), Val(ndims(buffer) - 1))...)

# Two-pool (MT) isochromat sequences hold a single `TwoPoolIsochromat` per voxel
# as well, so they share this kernel
function simulate_magnetization!(magnetization, ::CUDALibs, sequence::Union{IsochromatSimulator,TwoPoolIsochromatSimulator}, parameters)

    function kernel!(buffer, sequence, parameters, offset)
        i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        i > size(buffer, 1) && return nothing
        m = initialize_states(CUDALibs(), sequence)
        simulate_magnetization!(_voxel_output(buffer, i), sequence, m, @inbounds parameters[offset+i])
        return nothing
    end

    _in_voxel_chunks((magnetization,)) do (buffer,), offset
        nr_blocks = cld(size(buffer, 1), THREADS_PER_BLOCK)
        CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK kernel!(buffer, sequence, parameters, offset)
    end
    return nothing
end

function _simulate!(magnetization, ∂magnetization::NamedTuple{derivatives}, ::CUDALibs, sequence::IsochromatSimulator, parameters) where {derivatives}

    function kernel!(buffer, ∂buffers, sequence, parameters, offset)
        i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        i > size(buffer, 1) && return nothing
        m = initialize_derivative_states(CUDALibs(), sequence, Val(derivatives))
        output = (_voxel_output(buffer, i), tangents_of(map(∂b -> _voxel_output(∂b, i), ∂buffers)))
        simulate_magnetization!(output, sequence, m, @inbounds parameters[offset+i])
        return nothing
    end

    _in_voxel_chunks((magnetization, values(∂magnetization)...)) do (buffer, ∂buffers...), offset
        ∂buffers = NamedTuple{derivatives}(∂buffers)
        nr_blocks = cld(size(buffer, 1), THREADS_PER_BLOCK)
        CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK kernel!(buffer, ∂buffers, sequence, parameters, offset)
    end
    return nothing
end
