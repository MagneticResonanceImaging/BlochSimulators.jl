# Forward sensitivity for the two-pool (MT) EPG model
# =====================================================================
# Extends forward_sensitivity.jl to the two-pool EPG sequences
# (`TwoPoolEPGSimulator`, e.g. MTFISP2D), with the tangent sets ∂Ω∂MT and
# ∂Ω∂MTB₁ defined there. A two-pool simulation carries, besides the free pool's
# configuration states Ω, the bound pool's per-order population Zᵇ (see
# src/operators/epg.jl). Both become `(value, tangents)` pairs: Ω as everywhere
# else, Zᵇ as an SVector with one SVector tangent per property.
#
# The free pool's operators (excite!, dephasing!, sample_transverse!, E₂) are the
# ones in forward_sensitivity.jl. On top of those, the MT properties enter in
# three places only, each of which gets a source term here:
#
#   exchange_propagator    Λ and c depend on T₁, T₁ᵇ, k and f
#   lineshape              Wτ depends on T₂ᵇ, through the lineshape
#   mt_initial_conditions! the equilibrium (1-f, f) depends on f
#
# B₁ only enters through the free pool's excite! (the bound pool's saturation uses
# the nominal flip angle), so the B₁ tangent of Zᵇ has no source term.
#
# Limitation: at exactly f = 0 there is no bound pool and exchange is switched off,
# so the derivatives w.r.t. T₁ᵇ, T₂ᵇ and k are (correctly) zero, but the one w.r.t.
# f is not the limit f → 0⁺: the reverse rate k(1-f)/f has no derivative there. It
# only contains the f-dependence of the equilibrium. Use finite differences (which
# step to f > 0) at f = 0.

const S_MT{A} = Union{Tuple{A,<:∂Ω∂MT},Tuple{A,<:∂Ω∂MTB₁}}

# Rebuild a tangent set of the same kind (with or without B₁) from its fields.
@inline _MT_tangents(∂::∂Ω∂MT, ∂T₁, ∂T₂, ∂B₁, ∂T₁ᵇ, ∂T₂ᵇ, ∂f, ∂k) = ∂Ω∂MT(∂T₁, ∂T₂, ∂T₁ᵇ, ∂T₂ᵇ, ∂f, ∂k)
@inline _MT_tangents(∂::∂Ω∂MTB₁, ∂T₁, ∂T₂, ∂B₁, ∂T₁ᵇ, ∂T₂ᵇ, ∂f, ∂k) = ∂Ω∂MTB₁(∂T₁, ∂T₂, ∂B₁, ∂T₁ᵇ, ∂T₂ᵇ, ∂f, ∂k)

# Exchange propagator
# =====================================================================
# Λ = exp(Δt ΛL) and c = (I - Λ) Meq with Meq = (1-f, f). The derivative of the
# matrix exponential in a direction E = ∂ΛL/∂θ follows from the same eigen-
# decomposition used for Λ itself (Daleckii-Krein): with the spectral projectors
# P₁, P₂ of ΛL,
#
#   ∂Λ = Σᵢⱼ φᵢⱼ Pᵢ E Pⱼ,   φᵢᵢ = Δt e^{λᵢΔt},   φ₁₂ = (e^{λ₁Δt} - e^{λ₂Δt}) / (λ₁ - λ₂)
#
# φ₁₂ is evaluated as e^{λ₁Δt} (1 - e^{-(λ₁-λ₂)Δt}) / (λ₁ - λ₂) with expm1, which
# neither cancels for λ₁ ≈ λ₂ nor overflows for a very fast λ₂ (small f). In the
# degenerate case the value falls back to a first-order expansion around ΛL = λI,
# and so does this: ∂Λ = Δt e^{λΔt} E.

@inline function _∂exp(Δt, ΛL, λ₁, λ₂, E)
    T = typeof(λ₁)
    eΔtλ₁, eΔtλ₂ = exp(Δt * λ₁), exp(Δt * λ₂)
    λ₁ ≈ λ₂ && return Δt * eΔtλ₁ * E
    I₂ = SMatrix{2,2,T}(1, 0, 0, 1)
    P₁ = (ΛL - λ₂ * I₂) / (λ₁ - λ₂)
    P₂ = (ΛL - λ₁ * I₂) / (λ₂ - λ₁)
    φ₁₂ = eΔtλ₁ * -expm1(-(λ₁ - λ₂) * Δt) / (λ₁ - λ₂)
    return Δt * eΔtλ₁ * P₁ * E * P₁ + Δt * eΔtλ₂ * P₂ * E * P₂ + φ₁₂ * (P₁ * E * P₂ + P₂ * E * P₁)
end

"""
    exchange_propagator((Ω, ∂Ω)::S_MT, Δt, T₁ᵃ, T₁ᵇ, k, f)

Forward-sensitivity version of [`exchange_propagator`](@ref): returns `(Λ, ∂Λ)` and
`(c, ∂c)`, where `∂Λ` and `∂c` are `NamedTuple`s with the derivatives w.r.t. `T₁`,
`T₁ᵇ`, `f` and `k`, the only properties Λ and c depend on.
"""
@inline function exchange_propagator(::S_MT, Δt::T, T₁ᵃ::T, T₁ᵇ::T, k::T, f::T) where {T}
    Λ, c = exchange_propagator(Δt, T₁ᵃ, T₁ᵇ, k, f)
    ΛL, λ₁, λ₂ = _exchange_rate_matrix(T₁ᵃ, T₁ᵇ, k, f)

    # ∂ΛL/∂θ, with ΛL = [-R₁ᵃ-kᵃᵇ kᵇᵃ; kᵃᵇ -R₁ᵇ-kᵇᵃ], R₁ = 1/T₁, kᵃᵇ = k and
    # kᵇᵃ = k(1-f)/f, except at f = 0 where both rates are zero (see the limitation
    # at the top of this file)
    O = zero(T)
    E_T₁ = SMatrix{2,2,T}(1 / T₁ᵃ^2, O, O, O)
    E_T₁ᵇ = SMatrix{2,2,T}(O, O, O, 1 / T₁ᵇ^2)
    r, q = iszero(f) ? (O, O) : ((1 - f) / f, k / f^2)
    E_k = iszero(f) ? zero(SMatrix{2,2,T}) : SMatrix{2,2,T}(-1, 1, r, -r)
    E_f = SMatrix{2,2,T}(O, O, -q, q)

    ∂Λ = (T₁=_∂exp(Δt, ΛL, λ₁, λ₂, E_T₁), T₁ᵇ=_∂exp(Δt, ΛL, λ₁, λ₂, E_T₁ᵇ),
        f=_∂exp(Δt, ΛL, λ₁, λ₂, E_f), k=_∂exp(Δt, ΛL, λ₁, λ₂, E_k))

    # c = (I - Λ) Meq, so ∂c = -∂Λ Meq + (I - Λ) ∂Meq, and only Meq = (1-f, f)
    # depends on f
    Meq = SVector{2,T}(1 - f, f)
    I₂ = SMatrix{2,2,T}(1, 0, 0, 1)
    ∂c = (T₁=-∂Λ.T₁ * Meq, T₁ᵇ=-∂Λ.T₁ᵇ * Meq,
        f=-∂Λ.f * Meq + (I₂ - Λ) * SVector{2,T}(-1, 1), k=-∂Λ.k * Meq)

    return (Λ, ∂Λ), (c, ∂c)
end

# RF saturation
# =====================================================================
# Wτ = π ω₁²τ g(Δ, T₂ᵇ), so ∂Wτ/∂T₂ᵇ = π ω₁²τ ∂g/∂T₂ᵇ. The lineshapes all have
# the form g = T₂ᵇ h(Δ T₂ᵇ), differentiated term by term below.

@inline function lineshape_and_derivative(Δ::T, T₂ᵇ::T, ls::Gaussian) where {T}
    g = lineshape(Δ, T₂ᵇ, ls)
    a = 2 * T(π) * Δ
    return g, g * (1 / T₂ᵇ - a^2 * T₂ᵇ)
end

@inline function lineshape_and_derivative(Δ::T, T₂ᵇ::T, ls::Lorentzian) where {T}
    x² = (2 * T(π) * Δ * T₂ᵇ)^2
    return lineshape(Δ, T₂ᵇ, ls), (1 - x²) / (T(π) * (1 + x²)^2)
end

# Each quadrature term g = c T₂ᵇ exp(-2 (a T₂ᵇ / d)²) has ∂g/∂T₂ᵇ = g (1/T₂ᵇ - 4 a² T₂ᵇ / d²)
@inline function _superlorentzian_raw_and_derivative(Δ::T, T₂ᵇ::T) where {T}
    n = 500
    du = one(T) / (n - 1)
    a = 2 * T(π) * Δ
    acc, ∂acc = zero(T), zero(T)
    for i in 0:n-1
        u = i * du
        denom = 3 * u^2 - one(T)
        g = sqrt(2 / T(π)) * T₂ᵇ / abs(denom)
        g *= exp(-2 * (a * T₂ᵇ / denom)^2)
        acc += g
        ∂acc += g * (1 / T₂ᵇ - 4 * a^2 * T₂ᵇ / denom^2)
    end
    return acc * du, ∂acc * du
end

@inline function lineshape_and_derivative(Δ::T, T₂ᵇ::T, ::SuperLorentzian) where {T}
    Δc, Δo = T(1500), T(2000)   # as in `lineshape(Δ, T₂ᵇ, ::SuperLorentzian)`
    abs(Δ) >= Δc && return _superlorentzian_raw_and_derivative(Δ, T₂ᵇ)
    gc, ∂gc = _superlorentzian_raw_and_derivative(Δc, T₂ᵇ)
    go, ∂go = _superlorentzian_raw_and_derivative(Δo, T₂ᵇ)
    return _superlorentzian_interpolate(Δ, gc, go, Δc, Δo), _superlorentzian_interpolate(Δ, ∂gc, ∂go, Δc, Δo)
end

# The sequence evaluates the lineshape once per voxel, with the state as first
# argument; here that returns `(g, ∂g∂T₂ᵇ)`, which the per-pulse power scaling of
# `saturation_exponent` then turns into `(Wτ, ∂Wτ∂T₂ᵇ)`.
@inline lineshape(::S_MT, Δ, T₂ᵇ, ls::MTLineshape) = lineshape_and_derivative(Δ, T₂ᵇ, ls)

@inline saturation_exponent(flip::T, τ_RF::T, shape_factor::T, (g, ∂g∂T₂ᵇ)::Tuple{T,T}) where {T} =
    (saturation_exponent(flip, τ_RF, shape_factor, g), saturation_exponent(flip, τ_RF, shape_factor, ∂g∂T₂ᵇ))

# Bound pool operators
# =====================================================================

# Both pools start at their equilibrium (1-f, f) at order 0, so only the f tangent
# starts nonzero, at (-1, 1).
@inline function mt_initial_conditions!((Ω, ∂Ω)::S_MT, (Zᵇ, ∂Zᵇ)::S_MT, f)
    initial_conditions!((Ω, ∂Ω))    # zeroes every tangent
    Zᵇ = mt_initial_conditions!(Ω, Zᵇ, f)
    C = eltype(Zᵇ)
    ∂Zᵇ∂f = _mt_add_equilibrium(∂Ω.∂f, zero(∂Zᵇ.∂f), SVector(C(-1), C(1)))
    O = zero(∂Zᵇ.∂T₁)
    return (Zᵇ, _MT_tangents(∂Zᵇ, O, O, has_B₁(∂Zᵇ) ? O : nothing, O, O, ∂Zᵇ∂f, O))
end

# Zᵇ e^{-Wτ}: every tangent is scaled alike, and the T₂ᵇ one picks up -∂Wτ Zᵇ e^{-Wτ}.
@inline function mt_saturate((Zᵇ, ∂Zᵇ)::S_MT, (Wτ, ∂Wτ∂T₂ᵇ))
    e = exp(-Wτ)
    ∂ = _MT_tangents(∂Zᵇ, ∂Zᵇ.∂T₁ * e, ∂Zᵇ.∂T₂ * e, has_B₁(∂Zᵇ) ? ∂Zᵇ.∂B₁ * e : nothing,
        ∂Zᵇ.∂T₁ᵇ * e, (∂Zᵇ.∂T₂ᵇ - ∂Wτ∂T₂ᵇ * Zᵇ) * e, ∂Zᵇ.∂f * e, ∂Zᵇ.∂k * e)
    return (Zᵇ * e, ∂)
end

# The transverse rows decay by E₂ (with T₂'s source term, as in decay!), and
# (Zᵃ, Zᵇ) of every order go through Λ, with source term ∂Λ (Zᵃ, Zᵇ) from the old
# state and ∂c added at order 0. `Zero()` for a property that Λ, c or E₂ does not
# depend on drops that term at compile time.

@inline _decay_transverse!(∂Ω, Ω, E₂, ::Zero) = (∂Ω .*= (E₂, E₂, one(E₂)); nothing)
@inline _decay_transverse!(∂Ω, Ω, E₂, ∂E₂) = (∂Ω .= (E₂, E₂, one(E₂)) .* ∂Ω .+ (∂E₂, ∂E₂, Zero()) .* Ω; nothing)

@inline _exchange_source(::Zero, zᵃ, zᵇ) = (Zero(), Zero())
@inline _exchange_source(∂Λ, zᵃ, zᵇ) = (∂Λ[1, 1] * zᵃ + ∂Λ[1, 2] * zᵇ, ∂Λ[2, 1] * zᵃ + ∂Λ[2, 2] * zᵇ)

@inline _add_equilibrium_tangent(∂Ω, ∂Zᵇ, ::Zero) = ∂Zᵇ
@inline _add_equilibrium_tangent(∂Ω, ∂Zᵇ, ∂c) = _mt_add_equilibrium(∂Ω, ∂Zᵇ, ∂c)

@inline function _mt_exchange_relax_tangent!(∂Ω, ∂Zᵇ::SVector{M,C}, Ω, Zᵇ, E₂ᵃ, ∂E₂ᵃ, Λ, ∂Λ, ∂c) where {M,C}
    _decay_transverse!(∂Ω, Ω, E₂ᵃ, ∂E₂ᵃ)
    ∂Zᵇnew = MVector{M,C}(undef)
    @inbounds for n in 1:size(Ω, 2)
        s₁, s₂ = _exchange_source(∂Λ, Ω[3, n], Zᵇ[n])
        ∂zᵃ, ∂zᵇ = ∂Ω[3, n], ∂Zᵇ[n]
        ∂Ω[3, n] = Λ[1, 1] * ∂zᵃ + Λ[1, 2] * ∂zᵇ + s₁
        ∂Zᵇnew[n] = Λ[2, 1] * ∂zᵃ + Λ[2, 2] * ∂zᵇ + s₂
    end
    return _add_equilibrium_tangent(∂Ω, SVector(∂Zᵇnew), ∂c)
end

@inline function mt_exchange_relax!((Ω, ∂Ω)::S_MT, (Zᵇ, ∂Zᵇ)::S_MT, (E₂ᵃ, ∂E₂ᵃ∂T₂), (Λ, ∂Λ), (c, ∂c))
    # Every tangent before the state, since the source terms read the old state.
    ∂T₁ = _mt_exchange_relax_tangent!(∂Ω.∂T₁, ∂Zᵇ.∂T₁, Ω, Zᵇ, E₂ᵃ, Zero(), Λ, ∂Λ.T₁, ∂c.T₁)
    ∂T₂ = _mt_exchange_relax_tangent!(∂Ω.∂T₂, ∂Zᵇ.∂T₂, Ω, Zᵇ, E₂ᵃ, ∂E₂ᵃ∂T₂, Λ, Zero(), Zero())
    ∂B₁ = has_B₁(∂Ω) ? _mt_exchange_relax_tangent!(∂Ω.∂B₁, ∂Zᵇ.∂B₁, Ω, Zᵇ, E₂ᵃ, Zero(), Λ, Zero(), Zero()) : nothing
    ∂T₁ᵇ = _mt_exchange_relax_tangent!(∂Ω.∂T₁ᵇ, ∂Zᵇ.∂T₁ᵇ, Ω, Zᵇ, E₂ᵃ, Zero(), Λ, ∂Λ.T₁ᵇ, ∂c.T₁ᵇ)
    ∂T₂ᵇ = _mt_exchange_relax_tangent!(∂Ω.∂T₂ᵇ, ∂Zᵇ.∂T₂ᵇ, Ω, Zᵇ, E₂ᵃ, Zero(), Λ, Zero(), Zero())
    ∂f = _mt_exchange_relax_tangent!(∂Ω.∂f, ∂Zᵇ.∂f, Ω, Zᵇ, E₂ᵃ, Zero(), Λ, ∂Λ.f, ∂c.f)
    ∂k = _mt_exchange_relax_tangent!(∂Ω.∂k, ∂Zᵇ.∂k, Ω, Zᵇ, E₂ᵃ, Zero(), Λ, ∂Λ.k, ∂c.k)
    Zᵇ = mt_exchange_relax!(Ω, Zᵇ, E₂ᵃ, Λ, c)
    return (Zᵇ, _MT_tangents(∂Zᵇ, ∂T₁, ∂T₂, ∂B₁, ∂T₁ᵇ, ∂T₂ᵇ, ∂f, ∂k))
end

# Allocating the states
# =====================================================================
# Each tangent is allocated exactly as an ordinary two-pool state (Ω, Zᵇ), spelled
# out one by one for the GPU's sake (see the notes in forward_sensitivity.jl).

@inline function initialize_derivative_states(resource, sequence::TwoPoolEPGSimulator, ::Val{MT_DERIVATIVES})
    s = initialize_states(resource, sequence)
    T₁ = initialize_states(resource, sequence)
    T₂ = initialize_states(resource, sequence)
    T₁ᵇ = initialize_states(resource, sequence)
    T₂ᵇ = initialize_states(resource, sequence)
    f = initialize_states(resource, sequence)
    k = initialize_states(resource, sequence)
    return (Ω=(s.Ω, ∂Ω∂MT(T₁.Ω, T₂.Ω, T₁ᵇ.Ω, T₂ᵇ.Ω, f.Ω, k.Ω)),
        Zᵇ=(s.Zᵇ, ∂Ω∂MT(T₁.Zᵇ, T₂.Zᵇ, T₁ᵇ.Zᵇ, T₂ᵇ.Zᵇ, f.Zᵇ, k.Zᵇ)))
end

@inline function initialize_derivative_states(resource, sequence::TwoPoolEPGSimulator, ::Val{MT_B₁_DERIVATIVES})
    s = initialize_states(resource, sequence)
    T₁ = initialize_states(resource, sequence)
    T₂ = initialize_states(resource, sequence)
    B₁ = initialize_states(resource, sequence)
    T₁ᵇ = initialize_states(resource, sequence)
    T₂ᵇ = initialize_states(resource, sequence)
    f = initialize_states(resource, sequence)
    k = initialize_states(resource, sequence)
    return (Ω=(s.Ω, ∂Ω∂MTB₁(T₁.Ω, T₂.Ω, B₁.Ω, T₁ᵇ.Ω, T₂ᵇ.Ω, f.Ω, k.Ω)),
        Zᵇ=(s.Zᵇ, ∂Ω∂MTB₁(T₁.Zᵇ, T₂.Zᵇ, B₁.Zᵇ, T₁ᵇ.Zᵇ, T₂ᵇ.Zᵇ, f.Zᵇ, k.Zᵇ)))
end
