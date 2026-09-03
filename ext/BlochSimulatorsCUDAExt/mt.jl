# GPU methods of the two-pool (MT) EPG operators in src/operators/epg.jl
# ----------------------------------------------------------------------
# The free pool's Ω is partitioned over the lanes of a warp exactly as for a
# single-pool `EPGSimulator` (see epg.jl in this folder), and the bound pool's
# per-order population Zᵇ follows the same partitioning.

"""
    initialize_states(::CUDALibs, sequence::TwoPoolEPGSimulator{T,Ns}) where {T,Ns}

GPU version of `initialize_states` for `TwoPoolEPGSimulator`. `Ω` uses the same
per-thread state-partitioning scheme as `initialize_states(::CUDALibs, ::EPGSimulator)`;
`Zᵇ` uses exactly the same per-thread column count.
"""
@inline function initialize_states(::CUDALibs, sequence::TwoPoolEPGSimulator{T,Ns}) where {T,Ns}
    if (Ns % WARPSIZE != 0)
        error("Number of states must be a multiple of the warp size (32)")
    end
    num_states_per_thread = Ns ÷ WARPSIZE
    Ω = ConfigurationStatesSubset(@SMatrix zeros(Ω_eltype(sequence), 3, num_states_per_thread))
    Zᵇ = zeros(SVector{num_states_per_thread,Ω_eltype(sequence)})
    return (Ω=Ω, Zᵇ=Zᵇ)
end

@inline function _mt_add_equilibrium(Ω::ConfigurationStatesSubset, Zᵇ::SVector{M,C}, c) where {M,C}
    if laneid() == 1
        @inbounds Ω[3, 1] += c[1]
        return Base.setindex(Zᵇ, Zᵇ[1] + c[2], 1)
    end
    return Zᵇ
end
