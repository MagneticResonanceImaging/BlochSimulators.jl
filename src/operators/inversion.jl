# Inversion models
# =====================================================================
# EPG sequences with an inversion prepulse (e.g. FISP3D) apply it as a single,
# instantaneous operation on the configuration states. Which operation is set
# by the inversion model the sequence holds:
#
#   IdealInversion                Z → -Z, whatever the tissue.
#   EffectiveAdiabaticInversion   Z → η Z, with η depending on T₁, T₂ and B₁, derived
#                                 from a Bloch simulation of the actual pulse.
#
# Both leave the transverse states alone, like `invert!(Ω)`, so the inversion
# should be followed by spoiling.

"""
    InversionModel

What an inversion prepulse does to the magnetization of an EPG sequence, applied
with `invert!(Ω, model, p)`. See [`IdealInversion`](@ref) and
[`EffectiveAdiabaticInversion`](@ref).
"""
abstract type InversionModel end

"""
    IdealInversion()

Perfect, instantaneous inversion: `Z → -Z` for states of all orders, whatever
the tissue properties. A B₁-insensitive (adiabatic) pulse without relaxation
during the pulse.
"""
struct IdealInversion <: InversionModel end

"""
    EffectiveAdiabaticInversion{T<:Real,N}

Effective, instantaneous approximation of an adiabatic inversion pulse that is
long enough for relaxation during the pulse to matter (typically ~10 ms). The
whole pulse is replaced by `Z → η Z` for states of all orders, with

    η = η₀(B₁) exp(-τ₁(B₁)/T₁ - τ₂(B₁)/T₂)

While the magnetization follows the effective field at an angle θ(t) from the
z-axis, it relaxes at rate `cos²θ/T₁ + sin²θ/T₂`, so `τ₁ ≈ ∫cos²θ dt` and
`τ₂ ≈ ∫sin²θ dt`. That relies on the magnetization following the effective
field, which is what makes the model specific to adiabatic pulses: for e.g. a
block pulse the T₂ loss is not exponential. `η₀` is the inversion without any
relaxation, which falls
short of -1 where the pulse is not fully adiabatic (e.g. at low B₁). All three
are tabulated on a uniform B₁ grid and interpolated with cubic Hermite
(Catmull-Rom) splines, so that the B₁ derivative is continuous. Outside the
grid the end values are used, so the B₁ derivative is zero there.

η accounts for everything that happens *during* the pulse. A sequence using
this model should therefore measure its inversion delay from the *end* of the
pulse, and a wait before the inversion should end at the *start* of the pulse.
`duration` is stored so that timing can be derived from it; the simulation
itself does not use it.

Neglected: T₁ regrowth towards equilibrium during the pulse, the part of the
T₁ loss that is not exponential, B₀ and the effect of the pulse on transverse
states. How much that matters depends on the pulse; compare
[`inversion_efficiency`](@ref) with `simulate_magnetization` of the pulse to
check. For a 9.5 ms hyperbolic secant pulse the difference is a few 10⁻³ of M₀
for T₁ ≥ 0.3 s, where an ideal inversion is off by up to ~0.5 (see the tests).

Construct with [`EffectiveAdiabaticInversion(pulse::AdiabaticPulse)`](@ref), which
fits the table to Bloch simulations of the pulse.

# Fields
- `duration::T`: Pulse duration in **seconds**
- `B₁_start::T`: First point of the B₁ grid (dimensionless)
- `ΔB₁::T`: Spacing of the B₁ grid (dimensionless)
- `η₀::SVector{N,T}`: Inversion efficiency without relaxation (dimensionless, -1 is perfect)
- `τ₁::SVector{N,T}`: Effective T₁ relaxation time during the pulse in **seconds**
- `τ₂::SVector{N,T}`: Effective T₂ relaxation time during the pulse in **seconds**
"""
struct EffectiveAdiabaticInversion{T<:Real,N} <: InversionModel
    duration::T
    B₁_start::T
    ΔB₁::T
    η₀::SVector{N,T}
    τ₁::SVector{N,T}
    τ₂::SVector{N,T}
end

"""
    EffectiveAdiabaticInversion(duration, B₁::AbstractRange, η₀, τ₁, τ₂)

An [`EffectiveAdiabaticInversion`](@ref) from values `η₀`, `τ₁` and `τ₂` tabulated on the uniform
grid `B₁`, e.g. read back from a file. See the struct for the meaning and units.
"""
function EffectiveAdiabaticInversion(duration, B₁::AbstractRange, η₀::AbstractVector, τ₁::AbstractVector, τ₂::AbstractVector)
    N = length(B₁)
    N >= 2 || throw(ArgumentError("the B₁ grid needs at least two points"))
    length(η₀) == length(τ₁) == length(τ₂) == N || throw(ArgumentError(
        "η₀, τ₁ and τ₂ must have one value per point of the B₁ grid ($N)"))
    T = float(promote_type(typeof(duration), eltype(B₁), eltype(η₀), eltype(τ₁), eltype(τ₂)))
    return EffectiveAdiabaticInversion{T,N}(duration, first(B₁), step(B₁),
        SVector{N,T}(η₀), SVector{N,T}(τ₁), SVector{N,T}(τ₂))
end

# To be able to change precision. The fields are isbits, so the struct can be
# passed to a GPU kernel as part of the sequence as is.
@functor EffectiveAdiabaticInversion

Base.show(io::IO, model::EffectiveAdiabaticInversion{T,N}) where {T,N} = print(io,
    "EffectiveAdiabaticInversion{$T}($(round(1e3 * model.duration, sigdigits=4)) ms pulse, ",
    "$N B₁ points in [$(model.B₁_start), $(model.B₁_start + (N - 1) * model.ΔB₁)])")

"""
    inversion_efficiency(model::EffectiveAdiabaticInversion, p::AbstractTissueProperties)

The factor `η` by which `model` scales the longitudinal states, for tissue
properties `p`. Tissue properties without `B₁` are treated as `B₁ = 1`.
"""
@inline function inversion_efficiency(model::EffectiveAdiabaticInversion, p::AbstractTissueProperties)
    η₀, τ₁, τ₂ = _interpolate(model, _B₁(p))
    return η₀ * exp(-τ₁ / p.T₁ - τ₂ / p.T₂)
end

"""
    inversion_efficiency_and_derivatives(model::EffectiveAdiabaticInversion, p::AbstractTissueProperties)

`(η, ∂η/∂T₁, ∂η/∂T₂, ∂η/∂B₁)`, for forward sensitivity propagation. See
[`inversion_efficiency`](@ref).
"""
@inline function inversion_efficiency_and_derivatives(model::EffectiveAdiabaticInversion, p::AbstractTissueProperties)
    (η₀, τ₁, τ₂), (∂η₀, ∂τ₁, ∂τ₂) = _interpolate_with_slope(model, _B₁(p))
    T₁, T₂ = p.T₁, p.T₂
    relaxation = exp(-τ₁ / T₁ - τ₂ / T₂)
    η = η₀ * relaxation
    ∂η∂T₁ = η * τ₁ / T₁^2
    ∂η∂T₂ = η * τ₂ / T₂^2
    ∂η∂B₁ = relaxation * ∂η₀ - η * (∂τ₁ / T₁ + ∂τ₂ / T₂)
    return η, ∂η∂T₁, ∂η∂T₂, ∂η∂B₁
end

@inline _B₁(p::AbstractTissueProperties) = hasB₁(p) ? p.B₁ : one(p.T₁)

# Left grid point (1-based) of the interval that contains B₁, the position of
# B₁ within that interval (0 to 1), and whether B₁ lies inside the grid, i.e.
# whether the interpolant has a slope.
@inline function _grid_position(model::EffectiveAdiabaticInversion{T,N}, B₁) where {T,N}
    x = (B₁ - model.B₁_start) / model.ΔB₁
    inside = zero(x) <= x <= N - 1
    x = clamp(x, zero(x), oftype(x, N - 1))
    i = min(unsafe_trunc(Int, x), N - 2)
    return i + 1, x - i, inside
end

# Catmull-Rom spline through the tabulated values `v`, on the interval starting
# at grid point `i`: values at both ends of the interval plus the slopes there
# (central differences, one-sided at the ends of the grid), per grid step.
@inline function _hermite_data(v::SVector{N}, i) where {N}
    @inbounds begin
        vᵢ, vᵢ₊₁ = v[i], v[i+1]
        mᵢ = i > 1 ? (vᵢ₊₁ - v[i-1]) / 2 : vᵢ₊₁ - vᵢ
        mᵢ₊₁ = i + 1 < N ? (v[i+2] - vᵢ) / 2 : vᵢ₊₁ - vᵢ
    end
    return vᵢ, vᵢ₊₁, mᵢ, mᵢ₊₁
end

@inline function _hermite_value(v, i, w)
    vᵢ, vᵢ₊₁, mᵢ, mᵢ₊₁ = _hermite_data(v, i)
    w² = w * w
    w³ = w² * w
    return (2w³ - 3w² + 1) * vᵢ + (w³ - 2w² + w) * mᵢ + (3w² - 2w³) * vᵢ₊₁ + (w³ - w²) * mᵢ₊₁
end

# Derivative with respect to the position within the interval (so per grid step)
@inline function _hermite_slope(v, i, w)
    vᵢ, vᵢ₊₁, mᵢ, mᵢ₊₁ = _hermite_data(v, i)
    w² = w * w
    return (6w² - 6w) * (vᵢ - vᵢ₊₁) + (3w² - 4w + 1) * mᵢ + (3w² - 2w) * mᵢ₊₁
end

@inline function _interpolate(model::EffectiveAdiabaticInversion, B₁)
    i, w, _ = _grid_position(model, B₁)
    return _hermite_value(model.η₀, i, w), _hermite_value(model.τ₁, i, w), _hermite_value(model.τ₂, i, w)
end

@inline function _interpolate_with_slope(model::EffectiveAdiabaticInversion, B₁)
    i, w, inside = _grid_position(model, B₁)
    slope(v) = inside ? _hermite_slope(v, i, w) / model.ΔB₁ : zero(model.ΔB₁)
    return _interpolate(model, B₁), (slope(model.η₀), slope(model.τ₁), slope(model.τ₂))
end

"""
    invert!(Ω::AbstractConfigurationStates, model, p::AbstractTissueProperties)

Apply the inversion described by `model` ([`IdealInversion`](@ref) or
[`EffectiveAdiabaticInversion`](@ref)) to the longitudinal states of all orders.
*Assumes fully spoiled transverse magnetization*.
"""
@inline invert!(Ω::AbstractConfigurationStates, ::IdealInversion, p::AbstractTissueProperties) = invert!(Ω)

@inline function invert!(Ω::AbstractConfigurationStates, model::EffectiveAdiabaticInversion, p::AbstractTissueProperties)
    η = inversion_efficiency(model, p)
    Ω .*= (one(η), one(η), η)
end
