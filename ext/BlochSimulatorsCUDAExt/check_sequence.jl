# The GPU part of `check_sequence` (src/utils/check_sequence.jl): whether the
# (single precision) sequence can be moved to and simulated on the GPU, and
# whether the result agrees with the CPU.
function _check_gpu(sequence::BlochSimulator, parameters::AbstractVector, reference)
    CUDA.functional() || return nothing
    message = "runs on the GPU"
    name = nameof(typeof(sequence))

    sequence_gpu = try
        gpu(sequence)
    catch e
        return (false, message, "gpu(sequence) failed: " * sprint(showerror, e))
    end
    isbits(cudaconvert(sequence_gpu)) || return (false, message,
        "gpu(f32(sequence)) cannot be passed to a GPU kernel. Is the struct marked with `@adapt_structure $name`?")

    result = try
        Array(simulate_magnetization(CUDALibs(), sequence_gpu, gpu(parameters)))
    catch e
        # GPU compilation errors can be very long
        return (false, message, "Error: " * first(sprint(showerror, e), 2000))
    end
    difference = _relative_error(result, reference)
    return (difference < 1e-3, "runs on the GPU (relative difference with the CPU: $(_round(difference)))",
        "GPU and CPU, both in single precision, differ by more than 1e-3.")
end
