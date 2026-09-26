# GPU methods of the EPG operators in src/operators/epg.jl
# ---------------------------------------------------------
# On the GPU one warp simulates one voxel: each of its WARPSIZE lanes holds
# every WARPSIZE-th column of Ω in a `ConfigurationStatesSubset`, so lane 1
# holds state 0 and neighbouring states live on neighbouring lanes. The
# operators below are the ones for which that matters: whatever touches state
# 0 only runs on lane 1, and dephasing moves states between lanes with warp
# shuffles. Every other operator is shared with the CPU.

"""
    initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an array of EPG states on a CUDA GPU to be used throughout the simulation.
"""
@inline function initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}
    # # request shared memory in which configuration states are stored
    # # (all threads request for the entire threadblock)
    # Ω_shared = CUDA.CuStaticSharedArray(Ω_eltype(sequence), (3, Ns, THREADS_PER_BLOCK))
    # # get view for configuration states of this thread's voxel
    # # note that this function gets called inside a CUDA kernel
    # # so it has has access to threadIdx
    # Ω_view = view(Ω_shared, :, :, threadIdx().x)
    # # wrap in a ConfigurationStates object
    # Ω = MMatrix{3,Ns}(Ω_view) Ω = ConfigurationStates(Ω)

    # is Ns is not a multiple of 32, error
    if (Ns % WARPSIZE != 0)
        error("Number of states must be a multiple of the warp size (32)")
    end

    # Each thread holds (Ns ÷ WARPSIZE) columns of Ω
    #
    # Note this is stored as an *immutable* SMatrix, mutated by replacing the
    # `matrix` field of the (mutable) ConfigurationStatesSubset wrapper,
    # rather than as a mutable MMatrix mutated in place. MMatrix/MArray
    # implement setindex! via unsafe_store! through a pointer, which forces
    # the whole array into (very slow) GPU local memory. Once enough
    # operations in the per-TR loop touch it, the compiler ends up spilling
    # heavily -- this was the root cause of a >10x slowdown going from
    # max_state=32 to max_state=64. Reassigning an immutable SMatrix instead
    # keeps everything representable as plain SSA values, which the
    # compiler can keep in registers.
    num_states_per_thread = Ns ÷ WARPSIZE
    Ω = @SMatrix zeros(Ω_eltype(sequence), 3, num_states_per_thread)

    return ConfigurationStatesSubset(Ω)
end

# GPU version: only lane 1 owns global state 0 -- each lane holds a
# different strided subset of the states, so unlike CPU, not every lane
# should touch Z[0] (see initialize_states(::CUDALibs, ...)).
@inline function initial_conditions!(Ω::ConfigurationStatesSubset)
    fill!(Ω, zero(eltype(Ω)))
    if laneid() == 1
        @inbounds Ω[3, 1] = 1
    end
    return nothing
end

"""
On GPU, when each thread only holds parts of the configuration state matrix, each thread
only computes the corresponding columns of the diffusion decay matrix.
"""
@inline function diffusion_decay_matrix(Ω::ConfigurationStatesSubset, D::T) where {T<:Real}
    expbD = similar(real(Ω.matrix))
    for idx in 1:size(Ω, 2)
        # Calculate the state this thread is responsible for
        state = laneid() - 1 + (idx - 1) * WARPSIZE
        bᵀD = T(((state + 0.5)^2 + 1.0 / 12.0) * D)
        bᴸD = T((state^2) * D)
        expbᵀD = exp(-bᵀD)
        expbᴸD = exp(-bᴸD)
        expbD[1, idx] = expbᵀD
        expbD[2, idx] = expbᵀD
        expbD[3, idx] = expbᴸD
    end
    return ConfigurationStatesSubset(SMatrix(expbD))
end

# GPU version: same update as above, guarded to only lane 1 -- see the
# comment on initial_conditions!(Ω::ConfigurationStatesSubset) for why.
@inline function regrowth!(Ω::ConfigurationStatesSubset, E₁)
    if laneid() == 1
        @inbounds Ω[3, 1] += (1 - E₁)
    end
    return nothing
end

# GPU dephasing works directly on Ω. Its specialized setindex! above turns
# every apparent element mutation into a functional SMatrix update.
@inline function dephasing!(Ω::ConfigurationStatesSubset)
    shuffle_down!(Ω)
    shuffle_up!(Ω)
    return nothing
end

# shuffle down the F- states, set highest state to 0
@inline function shuffle_down!(Ω::ConfigurationStatesSubset)
    mask = CUDA.FULL_MASK
    src_lane = mod1(laneid() + 1, WARPSIZE)
    for column in axes(Ω, 2)
        @inbounds F̄₋ᵏ = Ω[2, column]
        if laneid() == Int32(1) && column < size(Ω, 2)
            @inbounds F̄₋ᵏ = Ω[2, column+1]
        end
        F̄₋ᵏ = CUDA.shfl_sync(mask, F̄₋ᵏ, src_lane)  # Broadcast value from the first lane
        @inbounds Ω[2, column] = F̄₋ᵏ
    end
    if laneid() == WARPSIZE
        @inbounds Ω[2, end] = 0
    end
    return nothing
end

# shuffle up the F₊ states and let F₊[0] be conj(F₋[0])
@inline function shuffle_up!(Ω::ConfigurationStatesSubset)
    mask = CUDA.FULL_MASK
    src_lane = mod1(laneid() - Int32(1), WARPSIZE)
    last_column = Int32(size(Ω, 2))
    for column in last_column:Int32(-1):Int32(1)
        @inbounds F₊ᵏ = Ω[1, column]
        if laneid() == WARPSIZE && column > Int32(1)
            @inbounds F₊ᵏ = Ω[1, column-1]
        end
        @inbounds Ω[1, column] = CUDA.shfl_sync(mask, F₊ᵏ, src_lane)
    end
    if laneid() == Int32(1)
        @inbounds Ω[1, 1] = conj(Ω[2, 1])
    end
    return nothing
end

@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::ConfigurationStatesSubset)
    if laneid() == 1
        @inbounds output[index] += Ω[1, 1]
    end
end

@inline function sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::ConfigurationStatesSubset)
    if laneid() == 1
        @inbounds output[index] .+= Ω
    end
end
