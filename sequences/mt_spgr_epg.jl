"""
    MTFISP2D{T,Ns,U<:AbstractVector,L<:MTLineshape} <: TwoPoolEPGSimulator{T,Ns}

Two-pool (free + bound pool) EPG simulator for a constant-TR, on-resonance
gradient-spoiled (FISP-type) sequence with magnetization transfer, following the binary
spin-bath model (see `operators/mt.jl`). RF pulses are instantaneous; the saturation of
the bound pool caused by each pulse is computed from its flip angle and duration via the
"equivalent CW power" approximation (see `saturation_exponent`), exactly the
approximation used by Malik et al.'s EPG-X (MRM 2018) `EPGX_GRE_MT`. Unlike the
two-pool isochromat model ([`MTSPGR2D`](@ref)), gradient spoiling here is exact
(analytic, via `dephasing!`) rather than emulated with an isochromat ensemble.

The magnetization is sampled directly after each excitation (`F₊[0]` right after the
RF pulse, matching `EPGX_GRE_MT`'s convention -- there is no separate echo-time delay).

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
- `max_state::Val{Ns}`: Maximum number of states to keep track of in EPG simulation
  (dimensionless).
"""
struct MTFISP2D{T,Ns,U<:AbstractVector,L<:MTLineshape} <: TwoPoolEPGSimulator{T,Ns}
    RF_train::U
    TR::T
    τ_RF::T
    pulse_shape_factor::T
    Δ::T
    lineshape::L
    max_state::Val{Ns}

    function MTFISP2D(RF_train::U, TR::T, τ_RF::T, pulse_shape_factor::T, Δ::T, lineshape::L, max_state::Val{Ns}) where {T,Ns,U<:AbstractVector,L<:MTLineshape}
        if mod(Ns, 32) != 0
            error("max_state must be a multiple of 32")
        end
        new{T,Ns,U,L}(RF_train, TR, τ_RF, pulse_shape_factor, Δ, lineshape, max_state)
    end
end

# To be able to change precision and send to CUDA device
@functor MTFISP2D
@adapt_structure MTFISP2D

export MTFISP2D

# Methods needed to allocate an output array of the correct size and type
output_size(sequence::MTFISP2D) = length(sequence.RF_train)
output_eltype(sequence::MTFISP2D{T}) where {T} = Complex{T}

# Sequence implementation
@inline function simulate_magnetization!(magnetization, sequence::MTFISP2D, state, p::AbstractTissueProperties)

    Ω, Zᵇ = state.Ω, state.Zᵇ
    TR = sequence.TR

    E₂ᵃ = E₂(Ω, TR, p.T₂)
    Λ, c = exchange_propagator(TR, p.T₁, p.T₁ᵇ, p.k, p.f)

    Zᵇ = mt_initial_conditions!(Ω, Zᵇ, p.f)

    @inbounds for (TRidx, RF) in enumerate(sequence.RF_train)

        # RF pulse: coherent rotation of the free pool's configuration states (row
        # 3 = Zᵃ included, exactly as for a single-pool sequence)...
        excite!(Ω, RF, p)

        # ...and saturation of the bound pool (independent of the above, spatially
        # uniform across all EPG orders).
        Wτ = saturation_exponent(abs(RF), sequence.τ_RF, sequence.pulse_shape_factor, sequence.Δ, p.T₂ᵇ, sequence.lineshape)
        Zᵇ = mt_saturate(Zᵇ, Wτ)

        # sample F₊[0] right after excitation
        sample_transverse!(magnetization, TRidx, Ω)

        # T₂ decay of the free pool, coupled T₁/exchange relaxation of (Zᵃ,Zᵇ)
        Zᵇ = mt_exchange_relax!(Ω, Zᵇ, E₂ᵃ, Λ, c)

        # shift F states due to the spoiling gradient (exact analytic spoiling)
        dephasing!(Ω)
    end

    return nothing
end

# Add method to getindex to reduce sequence length with convenient syntax
Base.getindex(seq::MTFISP2D, idx) = typeof(seq)(seq.RF_train[idx], seq.TR, seq.τ_RF, seq.pulse_shape_factor, seq.Δ, seq.lineshape, seq.max_state)

# Nicer printing of sequence information in REPL
Base.show(io::IO, seq::MTFISP2D) = begin
    println("")
    println(io, "MTFISP2D sequence")
    println(io, "RF_train:          ", typeof(seq.RF_train), " $(length(seq.RF_train)) flip angles (degrees)")
    println(io, "TR:                ", seq.TR, " s")
    println(io, "τ_RF:              ", seq.τ_RF, " s")
    println(io, "pulse_shape_factor:", seq.pulse_shape_factor)
    println(io, "Δ:                 ", seq.Δ, " Hz")
    println(io, "lineshape:         ", seq.lineshape)
    println(io, "max_state:         ", seq.max_state)
end

# Convenience constructor to quickly generate an MTFISP2D sequence of length nTR
MTFISP2D(nTR::Int) = MTFISP2D(complex.(ones(nTR) .* 15.0), 0.010, 0.001, 1.0, 0.0, SuperLorentzian(), Val(32))
