"""
    MTSPGR2D{T,Niso,U<:AbstractVector,L<:MTLineshape} <: TwoPoolIsochromatSimulator{T}

Two-pool (free + bound pool) isochromat simulator for a constant-TR, on-resonance
spoiled gradient-echo (SPGR) train with magnetization transfer, following the binary
spin-bath model (see `operators/mt.jl`). RF pulses are assumed instantaneous; the
saturation of the bound pool caused by each pulse is computed from its flip angle and
duration via the "equivalent CW power" approximation (see `saturation_exponent`), the
same approximation used by Malik et al.'s EPG-X (MRM 2018), including in *their own*
two-pool isochromat reference implementation.

Since a single on-axis isochromat cannot represent gradient spoiling by itself, spoiling
is emulated -- exactly as in EPG-X's isochromat reference -- by simulating an ensemble of
`Niso` isochromats per voxel, each accumulating a different multiple of `2π/Niso`
transverse phase per TR (`sequence.ψ`), and averaging their transverse magnetization.
This also makes it possible to check that the (in principle exact, for `Niso → ∞`)
isochromat result converges to the EPG result, which uses an analytic/infinite spoiling
assumption instead (see `scripts/validate_mt_epgx.jl`).

# Fields
- `RF_train::U`: Vector with flip angle for each TR. `abs.(RF_train)` are RF flip angles
  in **degrees** and `angle.(RF_train)` are RF phases in **radians**.
- `TR::T`: Repetition time in **seconds**, assumed constant during the sequence.
- `τ_RF::T`: Duration of each RF pulse, in **seconds** (used only to compute the bound
  pool's saturation, pulses are otherwise treated as instantaneous).
- `pulse_shape_factor::T`: Dimensionless RF pulse-shape factor (`1` for a
  rectangular/hard pulse); see `saturation_exponent`.
- `Δ::T`: Frequency offset of the RF pulses from the free pool's resonance, in **Hz**
  (`0` for on-resonance imaging pulses providing the only source of saturation).
- `lineshape::L`: Bound pool absorption lineshape, one of `Gaussian()`, `Lorentzian()`,
  `SuperLorentzian()`.
- `ψ::SVector{Niso,T}`: Per-TR transverse phase increment (**radians**) of each of the
  `Niso` isochromats in the spoiling-emulation ensemble.
"""
struct MTSPGR2D{T<:AbstractFloat,Niso,U<:AbstractVector{Complex{T}},L<:MTLineshape} <: TwoPoolIsochromatSimulator{T}
    RF_train::U
    TR::T
    τ_RF::T
    pulse_shape_factor::T
    Δ::T
    lineshape::L
    ψ::SVector{Niso,T}
end

# To be able to change precision and send to CUDA device
@functor MTSPGR2D
@adapt_structure MTSPGR2D

export MTSPGR2D

# Methods needed to allocate an output array of the correct size and type
output_size(sequence::MTSPGR2D) = length(sequence.RF_train)
output_eltype(sequence::MTSPGR2D{T}) where {T} = Complex{T}

# Sequence implementation
@inline function simulate_magnetization!(magnetization, sequence::MTSPGR2D{T}, m, p::AbstractTissueProperties) where {T}

    TR = sequence.TR

    # Precompute the TR-length T₂ decay factor of the free pool and the two-pool
    # relaxation-exchange propagator once (they don't change during the sequence).
    E₂ᵃ = exp(-TR / p.T₂)
    Λ, c = exchange_propagator(TR, p.T₁, p.T₁ᵇ, p.k, p.f)

    𝟘₃ = (zero(T), zero(T), zero(T))
    𝟘 = zero(T)

    @inbounds for ψᵢ in sequence.ψ

        m = initial_conditions(m, p)
        sinψ, cosψ = sincos(ψᵢ)

        for (TRidx, RF) in enumerate(sequence.RF_train)

            # RF pulse: coherent rotation of the free pool...
            γΔtRF = deg2rad(abs(RF)) * cis(angle(RF))
            m = rotate(m, γΔtRF, 𝟘₃, 𝟘₃, 𝟘, p)

            # ...and saturation of the bound pool (independent of the above).
            Wτ = saturation_exponent(abs(RF), sequence.τ_RF, sequence.pulse_shape_factor, sequence.Δ, p.T₂ᵇ, sequence.lineshape)
            m = saturate(m, Wτ)

            # sample transverse magnetization of the free pool right after excitation
            sample_transverse!(magnetization, TRidx, m)

            # spoiling gradient: increment transverse phase by ψᵢ. Averaged over the
            # ψ ensemble (see below), this emulates ideal gradient/RF spoiling.
            mᵃ = rotxy(sinψ, cosψ, Isochromat{T}(m.x, m.y, m.z))
            m = TwoPoolIsochromat{T}(mᵃ.x, mᵃ.y, mᵃ.z, m.zᵇ)

            # T₂ decay of the free pool, coupled T₁/exchange relaxation of (Zᵃ,Zᵇ)
            m = exchange_relax(m, E₂ᵃ, Λ, c)
        end
    end

    magnetization ./= length(sequence.ψ)

    return nothing
end

# Add method to getindex to reduce sequence length with convenient syntax
Base.getindex(seq::MTSPGR2D, idx) = typeof(seq)(seq.RF_train[idx], seq.TR, seq.τ_RF, seq.pulse_shape_factor, seq.Δ, seq.lineshape, seq.ψ)

# Nicer printing of sequence information in REPL
Base.show(io::IO, seq::MTSPGR2D) = begin
    println("")
    println(io, "MTSPGR2D sequence")
    println(io, "RF_train:          ", typeof(seq.RF_train), " $(length(seq.RF_train)) flip angles (degrees)")
    println(io, "TR:                ", seq.TR, " s")
    println(io, "τ_RF:              ", seq.τ_RF, " s")
    println(io, "pulse_shape_factor:", seq.pulse_shape_factor)
    println(io, "Δ:                 ", seq.Δ, " Hz")
    println(io, "lineshape:         ", seq.lineshape)
    println(io, "ψ:                 ", "SVector{$(length(seq.ψ))}{$(eltype(seq.ψ))} (radians)")
end

"""
    MTSPGR2D(RF_train, TR, τ_RF, pulse_shape_factor, Δ, lineshape, Niso::Int)

Convenience constructor that builds the `Niso`-isochromat spoiling-emulation ensemble
`ψ = 2π(0:Niso-1)/Niso` automatically.
"""
function MTSPGR2D(RF_train::AbstractVector{Complex{T}}, TR, τ_RF, pulse_shape_factor, Δ, lineshape::MTLineshape, Niso::Int) where {T}
    ψ = SVector{Niso,T}(ntuple(i -> 2 * T(π) * (i - 1) / Niso, Niso))
    return MTSPGR2D(RF_train, T(TR), T(τ_RF), T(pulse_shape_factor), T(Δ), lineshape, ψ)
end

# Convenience constructor to quickly generate an MTSPGR2D sequence of length nTR
MTSPGR2D(nTR::Int) = MTSPGR2D(complex.(ones(nTR) .* 15.0), 0.010, 0.001, 1.0, 0.0, SuperLorentzian(), 8)
