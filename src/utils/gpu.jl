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
    _on_gpu(sequence, parameters)

Whether the simulation of `sequence` for `parameters` should run on the GPU:
`true` when the parameters are on the GPU, `false` when they are not. Throws
if the sequence is on the other device. A sequence without arrays (other than
`StaticArray`s, which `gpu` leaves as they are) works on either.
"""
function _on_gpu(sequence, parameters)
    on_gpu = _arrays_on_gpu(parameters) === true
    sequence_on_gpu = _arrays_on_gpu(sequence)
    (sequence_on_gpu === nothing || sequence_on_gpu == on_gpu) ||
        throw(ArgumentError("Both sequence and parameters must be on the GPU or not on the GPU"))
    return on_gpu
end

"""
    _arrays_on_gpu(x)

Whether the arrays in `x` -- `x` itself, or its fields, recursively -- are on
the GPU: `true` if all of them are, `false` if not, and `nothing` if `x` holds
no arrays. `StaticArray`s are ignored, since `gpu` leaves them as they are.
"""
function _arrays_on_gpu(x)
    arrays = _collect_arrays!(Any[], x)
    return isempty(arrays) ? nothing : all(_is_gpu_array, arrays)
end

function _collect_arrays!(arrays, x)
    if x isa StaticArray || x isa Number
        # isbits, stays where it is
    elseif x isa AbstractArray && !(x isa StructArray)
        push!(arrays, x)
    else # a struct (or StructArray): look at its fields (or components)
        for name in propertynames(x)
            _collect_arrays!(arrays, getproperty(x, name))
        end
    end
    return arrays
end
