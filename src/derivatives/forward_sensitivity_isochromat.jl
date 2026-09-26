# Forward sensitivity propagation for isochromat-based sequences
# ----------------------------------------------------------------
# The isochromat counterpart of the EPG operators in forward_sensitivity.jl,
# which explains the approach. The simulation is handed the pair `(m, ∂m)` of
# an Isochromat and its tangents instead of a single Isochromat, and each
# operator in src/operators/isochromat.jl gets a method for that pair. The
# sequence's own `simulate_magnetization!` is then reused verbatim.
#
# The one difference with EPG is that the isochromat operators are functional
# -- `m = decay(m, E₁, E₂)` rather than `decay!(Ω, E₁, E₂)` -- so the methods
# below return a new pair instead of mutating one. The tangents are
# Isochromats themselves, which are small and immutable, so they are simply
# rebuilt, with `tangent_update`.
#
# The operators with a source term are the ones that depend on a tissue
# property:
#
#   decay, regrowth   E₁ and E₂ depend on T₁ and T₂
#   rotate (with RF)  B₁ scales the RF part of the rotation
#   invert(m, p)      the inversion angle is π B₁
#
# Rotations without RF, and the B₁-insensitive `invert(m)`, apply the same
# linear map to the state and to every tangent.

# The source term of a parameter that has none
@inline Base.:+(x::Isochromat, ::Zero) = x

"""
    tangent_update(f, ∂m, sources = (Zero(), Zero(), Zero()))

New tangents of an isochromat: `f` (the operator, as a function of the
isochromat) applied to each tangent, plus the source terms `(T₁, T₂, B₁)`.
The B₁ source is only used when a B₁ tangent is propagated.

`f` is typically a closure, which the compiler does not necessarily inline.
On the GPU a call that is not inlined returns its Isochromat through local
memory, so the calls are inlined explicitly, and so are the closures in
`rotate`. Measured for pSSFP3D with T₁ and T₂ derivatives: 296 bytes of local
memory per thread without either, 112 with both -- the same as a simulation
without derivatives.
"""
@inline tangent_update(f, ∂m::∂Ω∂T₁T₂, (s₁, s₂, _)=(Zero(), Zero(), Zero())) =
    ∂Ω∂T₁T₂(@inline(f(∂m.∂T₁)) + s₁, @inline(f(∂m.∂T₂)) + s₂)

@inline tangent_update(f, ∂m::∂Ω∂T₁T₂B₁, (s₁, s₂, s_B₁)=(Zero(), Zero(), Zero())) =
    ∂Ω∂T₁T₂B₁(@inline(f(∂m.∂T₁)) + s₁, @inline(f(∂m.∂T₂)) + s₂, @inline(f(∂m.∂B₁)) + s_B₁)

# Initial conditions: equilibrium, whatever the tissue properties

@inline initial_conditions((m, ∂m)::S{<:Isochromat}) =
    (initial_conditions(m), tangent_update(zero, ∂m))

# Relaxation
# =====================================================================
# decay scales (x, y, z) by (E₂, E₂, E₁), so by the product rule the T₁
# tangent gets a source in z and the T₂ tangent one in x and y. regrowth adds
# 1 - E₁ to z, so the T₁ tangent gets -∂E₁∂T₁ there.

@inline function decay((m, ∂m)::S{<:Isochromat}, (E₁, ∂E₁∂T₁), (E₂, ∂E₂∂T₂))
    T = eltype(m)
    ∂f∂T₁ = Isochromat{T}(0, 0, ∂E₁∂T₁ * m.z)
    ∂f∂T₂ = Isochromat{T}(∂E₂∂T₂ * m.x, ∂E₂∂T₂ * m.y, 0)
    ∂m = tangent_update(v -> decay(v, E₁, E₂), ∂m, (∂f∂T₁, ∂f∂T₂, Zero()))
    return (decay(m, E₁, E₂), ∂m)
end

@inline function regrowth((m, ∂m)::S{<:Isochromat}, (E₁, ∂E₁∂T₁))
    T = eltype(m)
    ∂m = tangent_update(identity, ∂m, (Isochromat{T}(0, 0, -∂E₁∂T₁), Zero(), Zero()))
    return (regrowth(m, E₁), ∂m)
end

# Rotations
# =====================================================================
# A rotation R acts on the state and every tangent alike. With RF, the
# rotation vector a has x and y components B₁ times the RF, so the B₁ tangent
# also picks up (∂R/∂B₁) m, the derivative of Rodrigues' formula along
# ∂a/∂B₁ (see `∂rodrigues`).

@inline function rotate((m, ∂m)::S{<:Isochromat}, γΔtRF, γΔtGR, r, Δt, p::AbstractTissueProperties, ΔtΔω)
    T = eltype(m)
    a = rotation_vector(T, γΔtRF, γΔtGR, r, Δt, p, ΔtΔω)
    θ = norm(a)
    ∂a∂B₁ = SVector{3,T}(real(γΔtRF), -imag(γΔtRF), 0)

    if iszero(θ)
        # No rotation, but the derivative of one is still defined: in the
        # limit θ → 0 Rodrigues' formula is m + a × m.
        source = has_B₁(∂m) ? Isochromat{T}(∂a∂B₁ × m) : Zero()
        return (m, tangent_update(identity, ∂m, (Zero(), Zero(), source)))
    end

    k = inv(θ) * a
    sinθ, cosθ = sincos(θ)
    @inline R(v) = rodrigues(v, k, sinθ, cosθ)
    source = has_B₁(∂m) ? ∂rodrigues(m, k, θ, sinθ, cosθ, ∂a∂B₁) : Zero()
    return (R(m), tangent_update(R, ∂m, (Zero(), Zero(), source)))
end

@inline rotate(state::S{<:Isochromat}, γΔtRF, γΔtGR, r, Δt, p::AbstractTissueProperties) =
    rotate(state, γΔtRF, γΔtGR, r, Δt, p, zero(eltype(first(state))))

"""
    ∂rodrigues(m, k, θ, sinθ, cosθ, δa)

Derivative of [`rodrigues`](@ref)`(m, k, sinθ, cosθ)` when its rotation vector
`a = θ k` changes by `δa`. Differentiating

    cosθ m + sinθ (k × m) + (k ⋅ m)(1 - cosθ) k

with `δθ = k ⋅ δa` and `δk = δa⊥ / θ`, where `δa⊥ = δa - δθ k` is the part of
`δa` that turns the axis. `δk` only appears multiplied by `sinθ` or
`1 - cosθ`, so those are divided by `θ` instead, which stays finite as θ → 0.
"""
@inline function ∂rodrigues(m::Isochromat{T}, k, θ, sinθ, cosθ, δa) where {T}
    δθ = k ⋅ δa
    δa⊥ = δa - δθ * k
    kᵀm = k ⋅ m
    ∂ = δθ * (-sinθ * m + cosθ * (k × m) + sinθ * kᵀm * k) +
        (sinθ / θ) * (δa⊥ × m) +
        ((one(T) - cosθ) / θ) * ((δa⊥ ⋅ m) * k + k⋅m * δa⊥)
    return Isochromat{T}(∂)
end

@inline function rotate((m, ∂m)::S{<:Isochromat}, γΔtGR, r, Δt, p::AbstractTissueProperties)
    sinθ, cosθ = sincos(precession_angle(γΔtGR, r, Δt, p))
    @inline R(v) = rotxy(sinθ, cosθ, v)
    return (R(m), tangent_update(R, ∂m))
end

# Inversion
# =====================================================================
# invert(m, p) sets z to cos(θ) z with θ = π B₁ (and x, y to 0), so the B₁
# tangent picks up -π sin(θ) z. The adiabatic invert(m) depends on nothing.

@inline function invert((m, ∂m)::S{<:Isochromat}, p::AbstractTissueProperties)
    T = eltype(m)
    source = if has_B₁(∂m)
        Isochromat{T}(0, 0, -π * sin(π * p.B₁) * m.z)
    else
        Zero()
    end
    return (invert(m, p), tangent_update(v -> invert(v, p), ∂m, (Zero(), Zero(), source)))
end

@inline invert((m, ∂m)::S{<:Isochromat}) = (invert(m), tangent_update(invert, ∂m))

# Sampling
# =====================================================================
# sample_transverse! is shared with the EPG model (forward_sensitivity.jl).

@inline function sample_xyz!((output, ∂output)::S, index, (m, ∂m)::S{<:Isochromat})
    sample_xyz!(output, index, m)
    sample_xyz!(∂output.∂T₁, index, ∂m.∂T₁)
    sample_xyz!(∂output.∂T₂, index, ∂m.∂T₂)
    has_B₁(∂m) && sample_xyz!(∂output.∂B₁, index, ∂m.∂B₁)
    return nothing
end

# Allocating the states
# =====================================================================

@inline function initialize_derivative_states(resource, sequence::IsochromatSimulator, ::Val{(:T₁, :T₂)})
    m = initialize_states(resource, sequence)
    return (m, ∂Ω∂T₁T₂(zero(m), zero(m)))
end

@inline function initialize_derivative_states(resource, sequence::IsochromatSimulator, ::Val{(:T₁, :T₂, :B₁)})
    m = initialize_states(resource, sequence)
    return (m, ∂Ω∂T₁T₂B₁(zero(m), zero(m), zero(m)))
end
