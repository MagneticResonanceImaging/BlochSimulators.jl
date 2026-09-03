# Two-pool magnetization transfer (MT) physics shared by the isochromat and EPG models.
#
# Follows the binary spin-bath model of Henkelman et al. (MRM 1993): a free pool `a`
# (what the rest of the package already simulates) exchanges *longitudinal*
# magnetization only with a semisolid bound pool `b` (T₂ᵇ is so short that the bound
# pool's transverse magnetization is never observable). RF pulses are assumed to act on
# the bound pool purely through saturation (an "equivalent CW power" approximation,
# Graham & Henkelman 1997 / Portnoy & Stanisz 2007), never through coherent rotation,
# following exactly the same convention used by Malik et al.'s EPG-X (MRM 2018).

### LINESHAPES ###

"""
    MTLineshape

Abstract type for the RF absorption lineshape of the bound pool. Subtypes are singleton
structs (zero runtime cost, dispatched on at compile time) so they can be stored as a
field of a `BlochSimulator` and safely moved to the GPU.
"""
abstract type MTLineshape end

"""
    Gaussian <: MTLineshape

Gaussian absorption lineshape for the bound pool.
"""
struct Gaussian <: MTLineshape end

"""
    Lorentzian <: MTLineshape

Lorentzian absorption lineshape for the bound pool.
"""
struct Lorentzian <: MTLineshape end

"""
    SuperLorentzian <: MTLineshape

Super-Lorentzian absorption lineshape for the bound pool (the standard choice for
tissue, e.g. Morrison & Henkelman 1995). The closed-form integral is singular near
`Δ = 0`; following Gloor et al. (MRM 2008) / Sled & Pike, values for `|Δ| < 1.5 kHz` are
obtained by interpolation (quadratic in `Δ²`, since the lineshape only depends on `Δ²`)
from values computed just outside the singular region, rather than by storing a lookup
table (which would not be voxel/GPU friendly since `T₂ᵇ` can vary per voxel).
"""
struct SuperLorentzian <: MTLineshape end

"""
    lineshape(Δ, T₂ᵇ, ::Gaussian)

Gaussian RF absorption lineshape value `g(Δ,T₂ᵇ)` (units: seconds).

# Arguments
- `Δ`: Frequency offset of the RF pulse from the free pool's resonance, in **Hz**.
- `T₂ᵇ`: Bound pool transverse relaxation time constant, in **seconds**.
"""
@inline function lineshape(Δ::T, T₂ᵇ::T, ::Gaussian) where {T}
    return T₂ᵇ / sqrt(2 * T(π)) * exp(-(2 * T(π) * Δ * T₂ᵇ)^2 / 2)
end

"""
    lineshape(Δ, T₂ᵇ, ::Lorentzian)

Lorentzian RF absorption lineshape value `g(Δ,T₂ᵇ)` (units: seconds).
"""
@inline function lineshape(Δ::T, T₂ᵇ::T, ::Lorentzian) where {T}
    return (T₂ᵇ / T(π)) / (1 + (2 * T(π) * Δ * T₂ᵇ)^2)
end

# Closed-form super-Lorentzian integral, evaluated with a fixed-order quadrature
# (mirrors the formula/quadrature order used in EPG-X's `SuperLorentzian.m`). This is
# only ever called once per voxel (to precompute a saturation rate, outside the hot
# per-TR loop), so the fixed 500-point sum is not a performance concern, and it is
# non-allocating so it also compiles fine inside a GPU kernel.
@inline function _superlorentzian_raw(Δ::T, T₂ᵇ::T) where {T}
    n = 500
    du = one(T) / (n - 1)
    acc = zero(T)
    for i in 0:n-1
        u = i * du
        denom = 3 * u^2 - one(T)
        g = sqrt(2 / T(π)) * T₂ᵇ / abs(denom)
        g *= exp(-2 * (2 * T(π) * Δ * T₂ᵇ / denom)^2)
        acc += g
    end
    return acc * du
end

"""
    lineshape(Δ, T₂ᵇ, ::SuperLorentzian)

Super-Lorentzian RF absorption lineshape value `g(Δ,T₂ᵇ)` (units: seconds), with
near-resonance (`|Δ| < 1.5 kHz`) interpolation to avoid the singularity at `Δ = 0`.
"""
@inline function lineshape(Δ::T, T₂ᵇ::T, ls::SuperLorentzian) where {T}
    Δc = T(1500)   # Hz, start of the singular region
    Δo = T(2000)   # Hz, second point used to estimate the trend into Δ=0
    if abs(Δ) >= Δc
        return _superlorentzian_raw(Δ, T₂ᵇ)
    else
        gc = _superlorentzian_raw(Δc, T₂ᵇ)
        go = _superlorentzian_raw(Δo, T₂ᵇ)
        # g(Δ) only depends on Δ², so interpolate/extrapolate linearly in Δ²
        slope = (go - gc) / (Δo^2 - Δc^2)
        g0 = gc - slope * Δc^2          # extrapolated value at Δ=0
        return g0 + (gc - g0) * (Δ^2 / Δc^2)
    end
end

### EXCHANGE-RELAXATION PROPAGATOR ###

"""
    exchange_propagator(Δt, T₁ᵃ, T₁ᵇ, k, f)

Analytic solution of the coupled two-pool longitudinal relaxation-exchange ODE

    d/dt [Zᵃ; Zᵇ] = Λ_L [Zᵃ; Zᵇ] + [R₁ᵃ(1-f); R₁ᵇf]

over a time interval `Δt`, computed in closed form via the eigen-decomposition of the
2×2 matrix `Λ_L` (no generic/allocating matrix exponential needed, unlike a naive
`exp(Matrix)`; this mirrors the `E₁`/`E₂` precomputation pattern used elsewhere in the
package -- call this once per sequence and reuse the result every relaxation interval).

# Arguments
- `Δt`: Time interval, in **seconds**.
- `T₁ᵃ`, `T₁ᵇ`: Longitudinal relaxation time constants of pool a (free) and pool b
  (bound), in **seconds**.
- `k`: Forward exchange rate a→b, in **s⁻¹**.
- `f`: Bound pool fraction of the total equilibrium magnetization (dimensionless).

# Returns
- `Λ::SMatrix{2,2}`: Propagator such that `[Zᵃ;Zᵇ](t+Δt) ≈ Λ*[Zᵃ;Zᵇ](t) + c`.
- `c::SVector{2}`: Equilibrium-regrowth contribution over `Δt`.
"""
@inline function exchange_propagator(Δt::T, T₁ᵃ::T, T₁ᵇ::T, k::T, f::T) where {T}

    R₁ᵃ, R₁ᵇ = inv(T₁ᵃ), inv(T₁ᵇ)
    # Reverse (b->a) rate from detailed balance; guard the f=0 (no bound pool) limit.
    kᵇᵃ = iszero(f) ? zero(T) : k * (1 - f) / f

    a11, a12 = -R₁ᵃ - k, kᵇᵃ
    a21, a22 = k, -R₁ᵇ - kᵇᵃ

    # Eigenvalues of ΛL (always real for this exchange-rate-matrix structure since
    # a12,a21 ≥ 0 and a11,a22 ≤ 0, so the discriminant (a11-a22)²+4a12a21 ≥ 0).
    τ = a11 + a22
    detΛL = a11 * a22 - a12 * a21
    disc = sqrt(max(τ^2 - 4 * detΛL, zero(T)))
    λ₁ = (τ + disc) / 2
    λ₂ = (τ - disc) / 2

    ΛL = SMatrix{2,2,T}(a11, a21, a12, a22)
    I₂ = SMatrix{2,2,T}(1, 0, 0, 1)

    eΔtλ₁, eΔtλ₂ = exp(Δt * λ₁), exp(Δt * λ₂)

    if λ₁ ≈ λ₂
        # Degenerate eigenvalues (measure-zero in practice): fall back to a first-order
        # (in λ₁-λ₂) expansion to avoid a 0/0 division.
        Λ = eΔtλ₁ * (I₂ + Δt * (ΛL - λ₁ * I₂))
    else
        P₁ = (ΛL - λ₂ * I₂) / (λ₁ - λ₂)
        P₂ = (ΛL - λ₁ * I₂) / (λ₂ - λ₁)
        Λ = eΔtλ₁ * P₁ + eΔtλ₂ * P₂
    end

    Meq = SVector{2,T}(1 - f, f)
    c = (I₂ - Λ) * Meq

    return Λ, c
end

### RF SATURATION ###

"""
    saturation_exponent(flip, τ_RF, shape_factor, Δ, T₂ᵇ, lineshape::MTLineshape)

Total (dimensionless) saturation `Wτ` of the bound pool caused by a single RF pulse,
using the "equivalent CW power" approximation (Graham & Henkelman 1997; Portnoy &
Stanisz, MRM 2007; matches the `WT` used in Malik et al.'s EPG-X). The bound pool's
magnetization is scaled by `exp(-Wτ)` at the moment of the pulse; the free pool is
unaffected by this (it undergoes its usual coherent rotation instead, see
`operators/isochromat.jl`/`operators/epg.jl`).

# Arguments
- `flip`: Nominal flip angle of the free pool for this pulse, in **degrees**.
- `τ_RF`: Pulse duration, in **seconds**.
- `shape_factor`: Dimensionless pulse-shape factor relating flip angle and duration to
  mean squared RF amplitude (`1` for a rectangular/hard pulse).
- `Δ`: Frequency offset of the pulse from the free pool's resonance, in **Hz**.
- `T₂ᵇ`: Bound pool transverse relaxation time constant, in **seconds**.
- `lineshape`: One of `Gaussian()`, `Lorentzian()`, `SuperLorentzian()`.
"""
@inline function saturation_exponent(flip::T, τ_RF::T, shape_factor::T, Δ::T, T₂ᵇ::T, ls::MTLineshape) where {T}
    ω₁rmsτ² = shape_factor * deg2rad(flip)^2 / τ_RF
    return T(π) * ω₁rmsτ² * lineshape(Δ, T₂ᵇ, ls)
end
