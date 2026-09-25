"""
    AdiabaticPulse{T<:Real, V<:AbstractVector} <: IsochromatSimulator{T}

This struct is used to simulate an adiabatic inversion pulse. This struct itself
could be used as field in other sequence structs.

# Fields
- `γΔtA::V`: Time-dependent amplitude modulation. Units: **radians** (γ·B₁·Δt)
- `Δω::V`: Time-dependent frequency modulation. Units: **rad/s** (angular frequency offset)
- `Δt::T`: Time discretization step in **seconds**, assumed constant
"""
struct AdiabaticPulse{T<:Real,V<:AbstractVector} <: IsochromatSimulator{T}
    γΔtA::V
    Δω::V
    Δt::T
end
# Methods needed to allocate an output array of the correct size and type
output_size(sequence::AdiabaticPulse) = 1 # Only record the magnetization at the end of the inversion pulse
output_eltype(sequence::AdiabaticPulse{T,V}) where {T,V} = Isochromat{T}

# To be able to change precision and send to CUDA device
@functor AdiabaticPulse
@adapt_structure AdiabaticPulse

## Sequence implementation
function simulate_magnetization!(output, sequence::AdiabaticPulse, m, p::AbstractTissueProperties)

    T₁, T₂ = p.T₁, p.T₂
    E₁, E₂ = BlochSimulators.E₁(m, sequence.Δt, T₁), BlochSimulators.E₂(m, sequence.Δt, T₂)

    m = initial_conditions(m)

    γΔtA, Δω, Δt = sequence.γΔtA, sequence.Δω, sequence.Δt

    𝟘 = zero(Δt)

    @inbounds for t in eachindex(sequence.γΔtA)

        m = rotate(m, γΔtA[t], 𝟘, 𝟘, Δt, p, Δt * Δω[t])
        m = decay(m, E₁, E₂)
        m = regrowth(m, E₁)

    end

    sample_xyz!(output, 1, m)
end

"""
    hyperbolic_secant_pulse(duration, B₁max; β = 4, μ = 5, Δt = 5e-6)

Hyperbolic secant adiabatic inversion pulse (Silver, Joseph & Hoult) as an
[`AdiabaticPulse`](@ref). With `t` running from -1 to 1 over the pulse:

    B₁(t) = B₁max sech(β t)
    Δf(t) = -f_max tanh(β t),   f_max = β μ / (π duration)

where `Δf` is the frequency offset of the pulse. `β` and `μ` are in units of the
half-duration, so the bandwidth-duration product is `2βμ/π` (≈ 12.7 for the
defaults). The sign of the frequency sweep does not matter on resonance, since
the pulse is symmetric.

# Arguments
- `duration`: Pulse duration in **seconds**
- `B₁max`: Peak B₁ amplitude in **Tesla**
- `β`, `μ`: Dimensionless shape parameters
- `Δt`: Target time step in **seconds**. Adjusted so that it divides `duration`.
"""
function hyperbolic_secant_pulse(duration, B₁max; β=4.0, μ=5.0, Δt=5e-6)
    γ = 2π * 42.577478e6 # rad/s/T
    nsteps = round(Int, duration / Δt)
    Δt = duration / nsteps
    # normalised time at the midpoint of each step, from -1 to 1
    t = [-1 + (2n - 1) / nsteps for n in 1:nsteps]
    f_max = β * μ / (π * duration)
    γΔtA = @. γ * B₁max * sech(β * t) * Δt
    # `Δω` is the off-resonance an on-resonant spin sees in the frame that rotates
    # along with the pulse's instantaneous frequency: -2π Δf.
    Δω = @. 2π * f_max * tanh(β * t)
    return AdiabaticPulse(γΔtA, Δω, Δt)
end

"""
    EffectiveAdiabaticInversion(pulse::AdiabaticPulse; B₁ = 0.4:0.02:1.6, T₁_ref = 1.0, T₂_ref = 0.05)

Fit an [`EffectiveAdiabaticInversion`](@ref) model to Bloch simulations of `pulse`. For
each B₁ on the grid, three simulations starting from equilibrium give

    η₀ = M_z(T₁ = ∞,     T₂ = ∞)
    τ₁ = -T₁_ref log(M_z(T₁ = T₁_ref, T₂ = ∞)      / η₀)
    τ₂ = -T₂_ref log(M_z(T₁ = ∞,      T₂ = T₂_ref) / η₀)

`τ₂` does not depend on the choice of `T₂_ref` (the T₂ loss is exponential).
The T₁ loss is not exactly exponential, so `τ₁` is fitted at a typical T₁; it
is only ~1% at T₁ = 1 s, so the choice matters little. See
[`EffectiveAdiabaticInversion`](@ref) for how close the result is to a Bloch simulation.

`B₁` must be a uniform range and should cover the B₁ values that will be
simulated; outside it the end values are used. Throws an `ArgumentError` if the
pulse does not invert at some B₁ of the grid (at low enough B₁ an adiabatic
pulse stops inverting), since the relaxation loss is then not defined.
"""
function EffectiveAdiabaticInversion(pulse::AdiabaticPulse{T}; B₁::AbstractRange=0.4:0.02:1.6, T₁_ref=1.0, T₂_ref=0.05) where {T}
    function Mz(T₁, T₂)
        parameters = StructVector(T₁T₂B₁{T}.(T₁, T₂, B₁))
        return map(m -> m.z, vec(simulate_magnetization(CPU1(), pulse, parameters)))
    end

    η₀ = Mz(Inf, Inf)
    loss₁ = Mz(T₁_ref, Inf) ./ η₀
    loss₂ = Mz(Inf, T₂_ref) ./ η₀

    # With η₀ near zero the pulse does not invert at all and the relaxation loss
    # is not a well-defined ratio.
    valid = @. (η₀ < -0.05) & (0 < loss₁ <= 1) & (0 < loss₂ <= 1)
    all(valid) || throw(ArgumentError(
        "the pulse does not invert (η₀ = $(round.(η₀[.!valid], digits=3))) at " *
        "B₁ = $(B₁[.!valid]); restrict the B₁ grid"))

    τ₁ = @. -T₁_ref * log(loss₁)
    τ₂ = @. -T₂_ref * log(loss₂)

    duration = length(pulse.γΔtA) * pulse.Δt
    return EffectiveAdiabaticInversion(duration, B₁, η₀, τ₁, τ₂)
end
