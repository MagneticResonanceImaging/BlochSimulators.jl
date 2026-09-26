"""
    check_sequence(sequence::BlochSimulator, parameters; io=stdout)

Check a sequence implementation against what BlochSimulators needs from it,
printing one line per check. Meant for custom sequences: run it once on a
few representative voxels while developing one. Pass `io=devnull` to only
get the result. `sequence` and `parameters`
(a vector of [`AbstractTissueProperties`](@ref), or a single one) should be
on the CPU, in double precision.

Checks, in order:
1. `output_size`, `output_eltype` and `simulate_magnetization!` are implemented.
2. The simulation runs on the CPU and its output is finite and not all zero.
3. `simulate_magnetization!` does not allocate, which is needed on the GPU.
4. `f32(sequence)` converts it to single precision (needs `@functor`), and
   single and double precision agree.
5. [`simulate_derivatives_forward_sensitivity`](@ref) agrees with finite
   differences. It fails when relaxation factors are computed directly (e.g.
   `exp(-Δt/T₁)`) rather than with `E₁(state, Δt, T₁)` and
   `E₂(state, Δt, T₂)`, which is what lets the derivatives be propagated.
6. When CUDA is loaded and a GPU is available: `gpu(f32(sequence))` can be
   passed to a kernel (needs `@adapt_structure`), and the simulation runs on
   the GPU and agrees with the CPU.

Checks 3 to 6 are skipped when check 1 or 2 fails.

Returns `true` if all checks passed (or were skipped).
"""
function check_sequence(sequence::BlochSimulator, parameters::AbstractVector{<:AbstractTissueProperties}; io::IO=stdout)

    passed = true
    report(ok, message, hint="") = begin
        passed &= ok !== false
        symbol, color = ok === nothing ? ("–", :light_black) : ok ? ("✓", :green) : ("✗", :red)
        printstyled(io, "  $symbol "; color, bold=true)
        println(io, message)
        (ok === false && !isempty(hint)) && printstyled(io, "    $hint\n"; color=:light_black)
    end
    println(io, "check_sequence($(nameof(typeof(sequence)))):")

    # 1. Interface
    missing_methods = String[]
    which(output_size, Tuple{typeof(sequence)}).sig == Tuple{typeof(output_size),BlochSimulator} && push!(missing_methods, "output_size")
    which(output_eltype, Tuple{typeof(sequence)}).sig == Tuple{typeof(output_eltype),BlochSimulator} && push!(missing_methods, "output_eltype")
    if isempty(missing_methods)
        state = initialize_states(CPU1(), sequence)
        magnetization = zeros(output_eltype(sequence), output_size(sequence)..., 1)
        output = selectdim(magnetization, ndims(magnetization), 1)
        signature = Tuple{typeof(output),typeof(sequence),typeof(state),eltype(parameters)}
        fallback = which(simulate_magnetization!, Tuple{Any,BlochSimulator,Any,AbstractTissueProperties})
        which(simulate_magnetization!, signature) === fallback && push!(missing_methods, "simulate_magnetization!")
    end
    report(isempty(missing_methods),
        "output_size, output_eltype and simulate_magnetization! are implemented",
        "Missing methods for $(nameof(typeof(sequence))): $(join(missing_methods, ", ")).")
    isempty(missing_methods) || return passed

    # 2. CPU simulation
    reference = try
        simulate_magnetization(CPU1(), sequence, parameters)
    catch e
        report(false, "simulates on the CPU", "Error: " * _short_error(e))
        return passed
    end
    finite = all(x -> all(isfinite, x), reference)
    nonzero = any(x -> !iszero(x), reference)
    report(finite && nonzero, "simulates on the CPU (output is finite and not all zero)",
        !finite ? "The output contains NaN or Inf." :
        "The output is all zero. Does simulate_magnetization! sample into `magnetization`, e.g. with sample_transverse!?")
    (finite && nonzero) || return passed

    # 3. Allocations
    bytes = _allocations(output, sequence, initialize_states(CPU1(), sequence), first(parameters))
    report(iszero(bytes), "simulate_magnetization! does not allocate",
        "It allocates $bytes bytes per voxel, which the GPU does not allow. Look for type " *
        "instabilities (@code_warntype) and for arrays created during the simulation.")

    # 4. Precision
    sequence32 = f32(sequence)
    if !(sequence32 isa BlochSimulator{Float32})
        report(false, "f32 converts the sequence to single precision",
            "f32(sequence) is not a BlochSimulator{Float32}. With Functors 0.4, is the struct " *
            "marked with `@functor $(nameof(typeof(sequence)))`? Does its type parameter T follow " *
            "from the types of its fields?")
    else
        error32 = try
            _relative_error(simulate_magnetization(CPU1(), sequence32, f32(parameters)), reference)
        catch e
            _short_error(e)
        end
        report(error32 isa Real && error32 < 1e-3,
            "f32 converts the sequence to single precision" *
            (error32 isa Real ? " (relative difference with double precision: $(_round(error32)))" : ""),
            error32 isa Real ? "Single and double precision differ by more than 1e-3." : "Error: $error32")
    end

    # 5. Forward-sensitivity derivatives
    p = first(parameters)
    derivatives = supported_derivatives(typeof(p))
    fd_error = try
        _, ∂m = simulate_derivatives_forward_sensitivity(derivatives, sequence, StructVector([p]))
        maximum(enumerate(derivatives)) do (i, θ)
            signal(x) = _flat(simulate_magnetization(CPU1(), sequence,
                StructVector([typeof(p)(setindex(SVector(Tuple(p)), x, i)...)])))
            h = iszero(p[i]) ? 1e-5 : 1e-5 * abs(p[i])
            fd = (-signal(p[i] + 2h) + 8signal(p[i] + h) - 8signal(p[i] - h) + signal(p[i] - 2h)) / (12h)
            norm(_flat(∂m[θ]) - fd) / (norm(fd) + 1e-10)
        end
    catch e
        _short_error(e)
    end
    report(fd_error isa Real && fd_error < 1e-4,
        "simulate_derivatives_forward_sensitivity agrees with finite differences" *
        (fd_error isa Real ? " (largest relative difference for $derivatives: $(_round(fd_error)))" : ""),
        (fd_error isa Real ? "" : "Error: $fd_error\n    ") *
        "Compute relaxation factors with E₁(state, Δt, T₁) and E₂(state, Δt, T₂), not exp(-Δt/T₁), " *
        "and use the operators from BlochSimulators for everything that depends on T₁, T₂ or B₁.")

    # 6. GPU
    gpu_result = _check_gpu(sequence32, f32(parameters), simulate_magnetization(CPU1(), sequence32, f32(parameters)))
    if gpu_result === nothing
        report(nothing, "GPU: skipped (load CUDA and have a functional GPU to check)")
    else
        ok, message, hint = gpu_result
        report(ok, message, hint)
    end

    return passed
end

check_sequence(sequence::BlochSimulator, p::AbstractTissueProperties; kwargs...) =
    check_sequence(sequence, StructVector([p]); kwargs...)

# Function barrier, so that only the simulation itself is measured
function _allocations(output, sequence, state, p)
    simulate_magnetization!(output, sequence, state, p)
    return @allocated simulate_magnetization!(output, sequence, state, p)
end

_relative_error(a, b) = norm(_flat(a) - _flat(b)) / norm(_flat(b))
_flat(x) = eltype(x) <: Isochromat ? reinterpret(eltype(eltype(x)), vec(x)) : vec(x)
_round(x) = round(x, sigdigits=2)

# The first line of the error message, of the error inside a failed task
function _short_error(e)
    e isa TaskFailedException && return _short_error(e.task.exception)
    e isa CompositeException && return _short_error(first(e.exceptions))
    return first(split(sprint(showerror, e), '\n'))
end

# GPU check. Returns `nothing` when it cannot run, and otherwise
# `(passed, message, hint)`. The method that does the work is in the CUDA
# extension; this one applies while CUDA is not loaded.
_check_gpu(sequence, parameters, reference) = nothing

export check_sequence
