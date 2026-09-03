"""
    AbstractTissueProperties{N,T} <: FieldVector{N,T}

Abstract type for custom structs that hold tissue properties used for a
simulation within one voxel. For simulations, `SimulationParameters`s are used
that can be assembled with the `@parameters` macro.

# Possible fields:
- `T₁::T`: Longitudinal relaxation time constant in **seconds**.
- `T₂::T`: Transverse relaxation time constant in **seconds**.
- `B₁::T`: Relative transmit B₁ field scaling factor (dimensionless, typically around 1.0).
- `B₀::T`: Off-resonance frequency in **Hz** (Hertz).
- `D::T`:  Diffusion coefficient in **m²/s**. The decay time constant T_D is calculated
            from D, TR, and the spoiling gradient area Δk in the sequence implementation.
- `ρˣ::T`: Real part of proton density or equilibrium magnetization M₀
  (arbitrary units, dimensionless scaling factor).
- `ρʸ::T`: Imaginary part of proton density or equilibrium magnetization M₀
  (arbitrary units, dimensionless scaling factor).
- `T₁ᵇ::T`: Longitudinal relaxation time constant of the semisolid/bound pool in a
  two-pool magnetization transfer (MT) model, in **seconds**.
- `T₂ᵇ::T`: Transverse relaxation time constant of the bound pool, in **seconds**
  (typically tens of microseconds). Used to evaluate the bound pool's RF absorption
  lineshape; the bound pool itself has no observable transverse magnetization.
- `f::T`: Bound pool fraction of the total equilibrium magnetization, i.e.
  `M0ᵇ / (M0ᵃ + M0ᵇ)` (dimensionless, between 0 and 1).
- `k::T`: Forward magnetization exchange rate from the free pool to the bound pool, in
  **s⁻¹**. The reverse rate follows from detailed balance: `k_ba = k * (1-f) / f`.

# Implementation details:
The structs are subtypes of FieldVector, which is a StaticVector with named
fields (see the documentation of StaticArrays.jl). There are three reasons for
letting the structs be subtypes of FieldVector:
1) FieldVectors/StaticVectors have sizes that are known at compile time. This is
   beneficial for performance reasons
2) The named fields improve readability of the code (e.g. `p.B₁` vs `p[3]`)
3) Linear algebra operations can be performed on instances of the structs. This
   allows, for example, subtraction (without having to manually define methods)
   and that is useful for comparing parameter maps.
"""
abstract type AbstractTissueProperties{N,T} <: FieldVector{N,T} end

# Define different TissueParameters types
"""
    T₁T₂{T} <: AbstractTissueProperties{2,T}

Tissue properties struct containing `T₁` and `T₂`. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂{T} <: AbstractTissueProperties{2,T}
    T₁::T
    T₂::T
end

"""
    T₁T₂B₁{T} <: AbstractTissueProperties{3,T}

Tissue properties struct containing `T₁`, `T₂`, and `B₁`. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₁{T} <: AbstractTissueProperties{3,T}
    T₁::T
    T₂::T
    B₁::T
end

"""
    T₁T₂B₀{T} <: AbstractTissueProperties{3,T}

Tissue properties struct containing `T₁`, `T₂`, and `B₀`. Units are defined in [`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₀{T} <: AbstractTissueProperties{3,T}
    T₁::T
    T₂::T
    B₀::T
end

"""
    T₁T₂B₁B₀{T} <: AbstractTissueProperties{4,T}

Tissue properties struct containing `T₁`, `T₂`, `B₁`, and `B₀`. Units are defined in [`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₁B₀{T} <: AbstractTissueProperties{4,T}
    T₁::T
    T₂::T
    B₁::T
    B₀::T
end

"""
    T₁T₂D{T} <: AbstractTissueProperties{3,T}

Tissue properties struct containing `T₁`, `T₂`, and `D`. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂D{T} <: AbstractTissueProperties{3,T}
    T₁::T
    T₂::T
    D::T
end

"""
    T₁T₂B₁D{T} <: AbstractTissueProperties{4,T}

Tissue properties struct containing `T₁`, `T₂`, `B₁`, and `D`. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₁D{T} <: AbstractTissueProperties{4,T}
    T₁::T
    T₂::T
    B₁::T
    D::T
end

"""
    T₁T₂B₀D{T} <: AbstractTissueProperties{5,T}

Tissue properties struct containing `T₁`, `T₂`, `B₀`, and `D`. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₀D{T} <: AbstractTissueProperties{4,T}
    T₁::T
    T₂::T
    B₀::T
    D::T
end

"""
    T₁T₂B₁B₀D{T} <: AbstractTissueProperties{5,T}

Tissue properties struct containing `T₁`, `T₂`, `B₁`, `B₀`, and `D`. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₁B₀D{T} <: AbstractTissueProperties{5,T}
    T₁::T
    T₂::T
    B₁::T
    B₀::T
    D::T
end

"""
    T₁T₂MT{T} <: AbstractTissueProperties{6,T}

Tissue properties struct containing `T₁`, `T₂`, `T₁ᵇ`, `T₂ᵇ`, `f`, and `k` for a two-pool
magnetization transfer (MT) model (free pool a + bound pool b, following the binary
spin-bath model of Henkelman et al.). Units are defined in [`AbstractTissueProperties`](@ref).
"""
struct T₁T₂MT{T} <: AbstractTissueProperties{6,T}
    T₁::T
    T₂::T
    T₁ᵇ::T
    T₂ᵇ::T
    f::T
    k::T
end

"""
    T₁T₂B₁MT{T} <: AbstractTissueProperties{7,T}

Tissue properties struct containing `T₁`, `T₂`, `B₁`, `T₁ᵇ`, `T₂ᵇ`, `f`, and `k` for a
two-pool magnetization transfer (MT) model. Units are defined in
[`AbstractTissueProperties`](@ref).
"""
struct T₁T₂B₁MT{T} <: AbstractTissueProperties{7,T}
    T₁::T
    T₂::T
    B₁::T
    T₁ᵇ::T
    T₂ᵇ::T
    f::T
    k::T
end

# For each subtype of AbstractTissueProperties created above, we use meta-programming to
# create additional types that also hold proton density (ρˣ and ρʸ).
#
# For example, given T₁T₂ <: AbstractTissueProperties, we automatically define:
#
# struct T₁T₂ρˣρʸ{T} <: AbstractTissueProperties{4,T}
#     T₁::T
#     T₂::T
#     ρˣ::T
#     ρʸ::T
# end
#
# as well as T₁T₂xy, T₁T₂, T₁T₂ρˣρʸ and T₁T₂ρˣρʸ.

for P in subtypes(AbstractTissueProperties)

    # Create new struct name by appending new fieldnames to name of struct
    structname_ρˣρʸ = Symbol(fieldnames(P)..., :ρˣρʸ)

    # Create new tuples of fieldnames
    fnames_ρˣρʸ = (fieldnames(P)..., :ρˣ, :ρʸ)

    fnames_typed_ρˣρʸ = [:($(fn)::T) for fn ∈ fnames_ρˣρʸ]

    N = length(fieldnames(P))
    # Format the field names string for the docstring outside the @eval block
    fnames_str = join(map(fn -> "`$fn`", fnames_ρˣρʸ), ", ")

    @eval begin
        """
            $($(structname_ρˣρʸ)){T} <: AbstractTissueProperties{$($(N+2)),T}

        Tissue properties struct containing $($(fnames_str)). Units are defined in [`AbstractTissueProperties`](@ref).
        """
        struct $(structname_ρˣρʸ){T} <: AbstractTissueProperties{$(N + 2),T}
            $(fnames_typed_ρˣρʸ...)
        end
    end
end

# The following is needed to make sure that operations with/on subtypes of AbstractTissueProperties return the appropriate type, see the FieldVector example from StaticArrays documentation
for S in subtypes(AbstractTissueProperties)
    @eval StaticArrays.similar_type(::Type{$(S){T}}, ::Type{T}, s::Size{(fieldcount($(S)),)}) where {T} = $(S){T}
end

# Define trait functions to check whether B₁, B₀, D or MT is part of the type
# Set default value to false:
hasB₁(::AbstractTissueProperties) = false
hasB₀(::AbstractTissueProperties) = false
hasD(::AbstractTissueProperties) = false
hasMT(::AbstractTissueProperties) = false

for P in subtypes(AbstractTissueProperties)
    @eval hasB₁(::$(P)) = $(:B₁ ∈ fieldnames(P))
    @eval hasB₀(::$(P)) = $(:B₀ ∈ fieldnames(P))
    @eval hasD(::$(P)) = $(:D ∈ fieldnames(P))
    @eval hasMT(::$(P)) = $(:f ∈ fieldnames(P))
end

# Programatically export all subtypes of AbstractTissueProperties
for P in subtypes(AbstractTissueProperties)
    @eval export $(Symbol(nameof(P)))
end

# Function to get the nonlinear part of the tissue properties
function get_nonlinear_part(p::Type{<:AbstractTissueProperties})
    parameter_set = fieldnames(p)
    if (:ρˣ ∉ parameter_set) && (:ρʸ ∉ parameter_set)
        return p
    elseif (:ρˣ ∈ parameter_set) && (:ρʸ ∈ parameter_set)

        if p <: T₁T₂ρˣρʸ
            return T₁T₂
        elseif p <: T₁T₂B₁ρˣρʸ
            return T₁T₂B₁
        elseif p <: T₁T₂Dρˣρʸ
            return T₁T₂D
        elseif p <: T₁T₂B₁Dρˣρʸ
            return T₁T₂B₁D
        elseif p <: T₁T₂B₀ρˣρʸ
            return T₁T₂B₀
        elseif p <: T₁T₂B₁B₀ρˣρʸ
            return T₁T₂B₁B₀
        elseif p <: T₁T₂B₁B₀Dρˣρʸ
            return T₁T₂B₁B₀D
        elseif p <: T₁T₂T₁ᵇT₂ᵇfkρˣρʸ
            return T₁T₂MT
        elseif p <: T₁T₂B₁T₁ᵇT₂ᵇfkρˣρʸ
            return T₁T₂B₁MT
        else
            error("Unknown parameter type: $p")
        end
    else
        error("Either both :ρˣ and :ρʸ should be included, or neither should be")
    end
end

"""
    macro parameters(args...)

Create a `SimulationParameters` with the actual struct type being determined by the arguments passed to the macro.

# Examples
```julia
# Create a StructArray{T₁T₂} with T₁ and T₂ values
T₁, T₂ = rand(100), 0.1*rand(100)
parameters = @parameters T₁ T₂

# Create a StructArray{T₁T₂B₁} with T₁, T₂ and B₁ values
T₁, T₂, B₁ = rand(100), 0.1*rand(100), ones(100)
parameters = @parameters T₁ T₂ B₁

# Create a StructArray{T₁T₂B₀} with T₁, T₂ and B₀ values
# This time use the aliases that don't use unicode characters
T1, T2, B0 = rand(100), 0.1*rand(100), ones(100)
parameters = @parameters T1 T2 B0
```
"""
macro parameters(args...)
    type_name = Symbol(join(string.(args)))
    first_arg_type = :(eltype($(esc(args[1]))))
    return :(StructArray{$(type_name){$(first_arg_type)}}(($(esc.(args)...),)))
end

# Define aliases for the tissue parameter types that do not use unicode characters such that, for example, `@parameters T1 T2 B0` is equivalent to `@parameters T₁ T₂ B₀`. This makes it easier for users of the package to generate tissue parameter arrays without having to type unicode characters.
const T1T2 = T₁T₂
const T1T2D = T₁T₂D
const T1T2B1 = T₁T₂B₁
const T1T2B1D = T₁T₂B₁D
const T1T2B0 = T₁T₂B₀
const T1T2B0D = T₁T₂B₀D
const T1T2B1B0 = T₁T₂B₁B₀
const T1T2B1B0D = T₁T₂B₁B₀D
const T1T2MT = T₁T₂MT
const T1T2B1MT = T₁T₂B₁MT

const T1T2PDxPDy = T₁T₂ρˣρʸ
const T1T2DPDxPDy = T₁T₂Dρˣρʸ
const T1T2B1PDxPDy = T₁T₂B₁ρˣρʸ
const T1T2B1DPDxPDy = T₁T₂B₁Dρˣρʸ
const T1T2B0PDxPDy = T₁T₂B₀ρˣρʸ
const T1T2B0DPDxPDy = T₁T₂B₀Dρˣρʸ
const T1T2B1B0PDxPDy = T₁T₂B₁B₀ρˣρʸ
const T1T2B1B0DPDxPDy = T₁T₂B₁B₀Dρˣρʸ
const T1T2MTPDxPDy = T₁T₂T₁ᵇT₂ᵇfkρˣρʸ
const T1T2B1MTPDxPDy = T₁T₂B₁T₁ᵇT₂ᵇfkρˣρʸ

# To perform simulations for multiple voxels, we store the tissue properties in a `StructArray` which we refer to as the `SimulationParameters`.
const SimulationParameters = StructArray{<:AbstractTissueProperties}
