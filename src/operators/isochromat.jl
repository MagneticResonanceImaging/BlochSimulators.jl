### TYPES ###

"""
    struct Isochromat{T<:Real} <: FieldVector{3,T}
        x::T
        y::T
        z::T
    end

Holds the x,y,z components of a spin isochromat in a FieldVector, which is a `StaticVector`
(from the package `StaticArrays`) with custom fieldnames.
"""
struct Isochromat{T<:Real} <: FieldVector{3,T}
    x::T
    y::T
    z::T
end

# Ensures that operations with/on Isochromats return Isochromats. See the FieldVector
# example from the StaticArrays documentation.
StaticArrays.similar_type(::Type{Isochromat{T}}, ::Type{T}, s::Size{(3,)}) where {T<:Real} = Isochromat{T}

### METHODS

# Initialize States

"""
    initialize_states(::AbstractResource, ::IsochromatSimulator{T}) where T

Initialize a spin isochromat to be used throughout a simulation of the sequence.

This may seem redundant but to is necessary to share the same programming interface with
`EPGSimulators`.
"""
@inline function initialize_states(::AbstractResource, ::IsochromatSimulator{T}) where {T}
    return Isochromat{T}(0, 0, 0)
end

# Initial conditions

"""
    initial_conditions(m::Isochromat{T}) where T

Return a spin isochromat with `(x,y,z) = (0,0,1)`.
"""
# Set initial conditions

@inline function initial_conditions(m::Isochromat{T}) where {T}
    return Isochromat{T}(0, 0, 1)
end

# Rotate

"""
    rotate(m::Isochromat{T}, γΔtRF, γΔtGR, r, Δt, p::AbstractTissueProperties, ΔtΔω=zero(T)) where T

RF, gradient and/or ΔB₀ induced rotation of Isochromat computed using Rodrigues' rotation
formula.

# Arguments
- `m`: Input isochromat state.
- `γΔtRF`: Complex value representing RF pulse effect (radians). `B₁` scaling from `p` is
  applied internally if `hasB₁(p)`.
- `γΔtGR`: Tuple/Vector representing gradient effect `γ * G * Δt` (radians/centimeter).
- `r`: Position vector `(x,y,z)` [cm].
- `Δt`: Time step duration (seconds).
- `p`: Tissue properties (`AbstractTissueProperties`). `B₀` effects from `p` are applied
  internally if `hasB₀(p)`.
- `ΔtΔω`: Additional off-resonance phase accumulated during `Δt` (radians). Often `Δt * Δω`,
  where `Δω` is in rad/s.

# Returns
- Rotated isochromat state.
"""
@inline function rotate(m::Isochromat{T}, γΔtRF, γΔtGR, r, Δt, p::AbstractTissueProperties, ΔtΔω=zero(T)) where {T}

    # Determine rotation vector a
    aˣ = real(γΔtRF)
    aʸ = -imag(γΔtRF)
    aᶻ = -(γΔtGR ⋅ r + ΔtΔω)

    hasB₁(p) && (aˣ *= p.B₁)
    hasB₁(p) && (aʸ *= p.B₁)
    hasB₀(p) && (aᶻ -= Δt * 2 * π * p.B₀)

    a = SVector{3,T}(aˣ, aʸ, aᶻ)

    # Angle of rotation is norm of rotation vector
    θ = norm(a)

    if !iszero(θ)
        # Normalize rotation vector
        k = inv(θ) * a
        # Perform rotation (Rodrigues formula)
        sinθ, cosθ = sincos(θ)
        m = (cosθ * m) + (sinθ * k × m) + (k ⋅ m * (one(T) - cosθ) * k)
    end

    return m
end

"""
    rotate(m::Isochromat, γΔtGRz, z, Δt, p::AbstractTissueProperties)

Rotation of Isochromat without RF (so around z-axis only) due to gradients and B0 (i.e.
refocussing slice select gradient).
"""
@inline function rotate(m::Isochromat, γΔtGR, r, Δt, p::AbstractTissueProperties)
    # Determine rotation angle θ
    θ = -γΔtGR ⋅ r
    hasB₀(p) && (θ -= Δt * π * p.B₀ * 2)
    # Perform rotation in xy plane
    sinθ, cosθ = sincos(θ)
    return rotxy(sinθ, cosθ, m)
end

# Decay

"""
    decay(m::Isochromat{T}, E₁, E₂) where T

Apply T₂ decay to transverse component and T₁ decay to longitudinal component of
`Isochromat`.
"""
@inline function decay(m::Isochromat{T}, E₁, E₂) where {T}
    return m .* Isochromat{T}(E₂, E₂, E₁)
end

# Regrowth

"""
    regrowth(m::Isochromat{T}, E₁) where T

Apply T₁ regrowth to longitudinal component of `Isochromat`.
"""
@inline function regrowth(m::Isochromat{T}, E₁) where {T}
    return m + Isochromat{T}(0, 0, 1 - E₁)
end

# Invert

"""
    invert(m::Isochromat{T}, p::AbstractTissueProperties) where T

Invert z-component of `Isochromat` (assuming spoiled transverse magnetization so
xy-component zero).
"""
@inline function invert(m::Isochromat{T}, p::AbstractTissueProperties) where {T}
    # Determine rotation angle θ
    θ = π
    hasB₁(p) && (θ *= p.B₁)
    return Isochromat{T}(0, 0, cos(θ) * m.z)
end

"""
    invert(m::Isochromat{T}, p::AbstractTissueProperties) where T

Invert `Isochromat` with B₁ insenstive (i.e. adiabatic) inversion pulse
"""
@inline invert(m::Isochromat{T}) where {T} = Isochromat{T}(0, 0, -m.z)

# Sample

"""
    sample!(output, index::Union{Integer,CartesianIndex}, m::Isochromat)

Sample transverse magnetization from `Isochromat`. The "+=" is needed for 2D sequences where
slice profile is taken into account.
"""
@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, m::Isochromat)
    @inbounds output[index] += complex(m.x, m.y)
end

"""
    sample_xyz!(output, index::Union{Integer,CartesianIndex}, m::Isochromat)

Sample m.x, m.y and m.z components from `Isochromat`. The "+=" is needed for 2D sequences
where slice profile is taken into account.
"""
@inline function sample_xyz!(output::AbstractArray{<:S}, index::Union{Integer,CartesianIndex}, m::Isochromat) where {S}
    @inbounds output[index] += S(m.x, m.y, m.z)
end

### TWO-POOL (MT) EXTENSION ###
#
# Extends the isochromat model with a second, bound pool (see `operators/mt.jl` for the
# underlying two-pool exchange/saturation physics). The bound pool only ever has a
# longitudinal component `zᵇ`, so the state is a 4-vector (x,y,z,zᵇ) rather than 3. RF
# and gradient rotation are unchanged for the free pool (x,y,z) -- reused directly from
# above -- and the bound pool is only ever affected through saturation, never rotation.

"""
    struct TwoPoolIsochromat{T<:Real} <: FieldVector{4,T}
        x::T
        y::T
        z::T
        zᵇ::T
    end

Holds the x,y,z components of the free pool's isochromat plus the longitudinal
magnetization `zᵇ` of the bound pool, for two-pool magnetization transfer simulations.
"""
struct TwoPoolIsochromat{T<:Real} <: FieldVector{4,T}
    x::T
    y::T
    z::T
    zᵇ::T
end

StaticArrays.similar_type(::Type{TwoPoolIsochromat{T}}, ::Type{T}, s::Size{(4,)}) where {T<:Real} = TwoPoolIsochromat{T}

# Initialize States

"""
    initialize_states(::AbstractResource, ::TwoPoolIsochromatSimulator{T}) where T

Initialize a two-pool spin isochromat to be used throughout a simulation of the sequence.
"""
@inline function initialize_states(::AbstractResource, ::TwoPoolIsochromatSimulator{T}) where {T}
    return TwoPoolIsochromat{T}(0, 0, 0, 0)
end

"""
    initial_conditions(m::TwoPoolIsochromat{T}, p::AbstractTissueProperties) where T

Return a two-pool isochromat with `(x,y,z) = (0,0,1)` for the free pool and `zᵇ = p.f`
for the bound pool (its equilibrium value).
"""
@inline function initial_conditions(m::TwoPoolIsochromat{T}, p::AbstractTissueProperties) where {T}
    return TwoPoolIsochromat{T}(0, 0, 1, p.f)
end

# Rotate

"""
    rotate(m::TwoPoolIsochromat, args...)

RF, gradient and/or ΔB₀ induced rotation of the free pool, identical to
[`rotate`](@ref) for a single-pool `Isochromat`. The bound pool has no transverse
component and is therefore never rotated (only saturated, see [`saturate`](@ref)); `zᵇ`
is passed through unchanged.
"""
@inline function rotate(m::TwoPoolIsochromat{T}, args...) where {T}
    mᵃ = rotate(Isochromat{T}(m.x, m.y, m.z), args...)
    return TwoPoolIsochromat{T}(mᵃ.x, mᵃ.y, mᵃ.z, m.zᵇ)
end

# Saturate

"""
    saturate(m::TwoPoolIsochromat{T}, Wτ) where T

Apply RF saturation to the bound pool, scaling `zᵇ` by `exp(-Wτ)` (see
[`saturation_exponent`](@ref)). The free pool is unaffected.
"""
@inline function saturate(m::TwoPoolIsochromat{T}, Wτ) where {T}
    return TwoPoolIsochromat{T}(m.x, m.y, m.z, m.zᵇ * exp(-Wτ))
end

# Exchange + relax

"""
    exchange_relax(m::TwoPoolIsochromat{T}, E₂ᵃ, Λ, c) where T

Apply T₂ decay to the free pool's transverse component (using `E₂ᵃ`, exactly like
[`decay`](@ref)) and the coupled two-pool relaxation-exchange propagator (`Λ`, `c` from
[`exchange_propagator`](@ref)) to `(z,zᵇ)` jointly. Replaces the `decay`+`regrowth` pair
used for single-pool sequences (T₁ relaxation of the free pool's z-component is no
longer independent of the bound pool, so it cannot be split into separate decay/regrowth
steps).
"""
@inline function exchange_relax(m::TwoPoolIsochromat{T}, E₂ᵃ, Λ, c) where {T}
    zᵃzᵇ = Λ * SVector{2,T}(m.z, m.zᵇ) + c
    return TwoPoolIsochromat{T}(m.x * E₂ᵃ, m.y * E₂ᵃ, zᵃzᵇ[1], zᵃzᵇ[2])
end

# Invert

"""
    invert(m::TwoPoolIsochromat{T}, p::AbstractTissueProperties, τ_RF, shape_factor, Δ, ls::MTLineshape) where T

Invert the free pool's z-component exactly like [`invert`](@ref) for a single-pool
`Isochromat`, and saturate the bound pool using the inversion pulse's own duration
`τ_RF` and lineshape parameters (an RF pulse only ever saturates -- never coherently
rotates -- the bound pool, whether it is an excitation or inversion pulse).
"""
@inline function invert(m::TwoPoolIsochromat{T}, p::AbstractTissueProperties, τ_RF, shape_factor, Δ, ls::MTLineshape) where {T}
    mᵃ = invert(Isochromat{T}(m.x, m.y, m.z), p)
    Wτ = saturation_exponent(T(180), τ_RF, shape_factor, Δ, p.T₂ᵇ, ls)
    return TwoPoolIsochromat{T}(mᵃ.x, mᵃ.y, mᵃ.z, m.zᵇ * exp(-Wτ))
end

"""
    invert(m::TwoPoolIsochromat{T}) where T

Invert the free pool with a B₁-insensitive (i.e. adiabatic) inversion pulse. As the
bound pool's response to an adiabatic sweep is not represented by the same "flip
angle + duration" saturation formula used for the free pool's imaging pulses, the bound
pool is left unchanged here; use the other `invert` method (with an explicit `τ_RF`) to
also saturate the bound pool.
"""
@inline invert(m::TwoPoolIsochromat{T}) where {T} = TwoPoolIsochromat{T}(0, 0, -m.z, m.zᵇ)

# Sample

"""
    sample_transverse!(output, index::Union{Integer,CartesianIndex}, m::TwoPoolIsochromat)

Sample transverse magnetization of the free pool from a `TwoPoolIsochromat` (the bound
pool has no transverse, observable magnetization).
"""
@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, m::TwoPoolIsochromat)
    @inbounds output[index] += complex(m.x, m.y)
end

"""
    sample_xyz!(output, index::Union{Integer,CartesianIndex}, m::TwoPoolIsochromat)

Sample m.x, m.y and m.z components of the free pool from a `TwoPoolIsochromat`.
"""
@inline function sample_xyz!(output::AbstractArray{<:S}, index::Union{Integer,CartesianIndex}, m::TwoPoolIsochromat) where {S}
    @inbounds output[index] += S(m.x, m.y, m.z)
end
