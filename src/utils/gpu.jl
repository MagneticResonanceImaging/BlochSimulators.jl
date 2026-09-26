# Convert non-isbits fields of structs (e.g. sequences, trajectories)
# to CuArrays.
# Inspired by KomaMRI.jl (KomaMRI.jl/src/simulation/GPUFunctions.jl),
# which in turn is inspired by Flux.jl (Flux.jl/src/functor.jl)
#
# CUDA.jl is a weak dependency: everything that needs it lives in
# ext/BlochSimulatorsCUDAExt, which Julia loads as soon as CUDA is loaded next
# to BlochSimulators (`using CUDA`). The two functions below are what the rest
# of the package calls to deal with GPU arrays. Their fallbacks here apply
# while CUDA is not loaded; the extension adds the methods for `CuArray`s.

"""
    _to_gpu_array(x::AbstractArray)

Copy `x` to the GPU. Only has a working method once CUDA is loaded.
"""
_to_gpu_array(x) = error(
    "Moving data to the GPU requires CUDA.jl: add `using CUDA` to your script. " *
    "BlochSimulators loads its GPU support once CUDA is loaded.")

"""
    _is_gpu_array(x)

Whether `x` is an array on the GPU. Always `false` while CUDA is not loaded.
"""
_is_gpu_array(x) = false

"""
    gpu(x)

Move `x` to CUDA device. It uses `Functors.fmap` to recursively
traverse the fields of the struct `x`, converting `<:AbstractArrays` to `CuArrays`,
and ignoring isbitsarrays. For custom structs (e.g. `<:BlochSimulator` or `<:AbstractTrajectory`),
it is required that `typeof(x)` be made a `Functors.@functor` (e.g. `@functor FISP`).

Requires CUDA.jl to be loaded (`using CUDA`).
"""
gpu(x) = fmap(_gpu, x; exclude=_isleaf)

_gpu(x) = x
_gpu(x::AbstractArray) = _to_gpu_array(x) # no automatic conversion to Float32
_gpu(x::AbstractTissueProperties) = x
_gpu(x::StaticArray) = x

# make custom _isleaf function to include isbitsarrays
_isbitsarray(::AbstractArray{<:Real}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false
_isleaf(x) = _isbitsarray(x) || isleaf(x)

# Convert underlying arrays of a `StructArray` to `CuArray`s while keeping the `StructArray` wrapper intact
_gpu(x::StructArray) = StructArray{eltype(x)}(gpu(StructArrays.components(x)))
_gpu(x::AbstractVector{<:StructArray}) = [gpu(x[i]) for i in eachindex(x)]

export gpu

"""
    _all_arrays_are_cuarrays(x)

Returns `true` if all `AbstractArray` fields in `x` are `CuArray`s and `false` otherwise. Will also `return` if `x` does not have any `AbstractArray` fields.
"""
function _all_arrays_are_cuarrays(x)
    for name in propertynames(x)
        property = getproperty(x, name)
        if (property isa AbstractArray) && !_is_gpu_array(property)
            return false
        end
    end
    return true
end
