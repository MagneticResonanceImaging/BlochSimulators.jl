# GPU methods of the simulation drivers in src/simulate/ and
# src/derivatives/forward_sensitivity.jl: output allocation and the CUDA
# kernels. Magnetization and derivative kernels assign one warp (WARPSIZE
# threads) to each voxel; the signal kernel one thread to each time point.

"""
However, if the `parameters` are a CuArray, the simulation is performed on the GPU.
"""
function simulate_magnetization(sequence, parameters::CuArray)
    simulate_magnetization(CUDALibs(), gpu(sequence), parameters)
end

function simulate_magnetization!(magnetization, ::CUDALibs, sequence, parameters)
    # GPU execution: Launch CUDA kernel with multiple threads per voxel
    # Each voxel gets assigned `WARPSIZE` threads for coalesced memory access
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

