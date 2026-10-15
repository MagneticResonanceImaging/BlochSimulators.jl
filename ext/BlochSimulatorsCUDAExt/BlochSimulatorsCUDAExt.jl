"""
GPU support for BlochSimulators, loaded automatically once CUDA.jl is loaded
next to BlochSimulators (`using CUDA`).
"""
module BlochSimulatorsCUDAExt

using BlochSimulators
using CUDA
using ComputationalResources: CUDALibs
using StaticArrays

using BlochSimulators: BlochSimulator, EPGSimulator, IsochromatSimulator, TwoPoolIsochromatSimulator, TwoPoolEPGSimulator, ConfigurationStatesSubset, Ω_eltype,
    initialize_derivative_states, tangents_of, signal_at_time_point!, nsamples,
    _relative_error, _round

# Functions that get GPU methods here
import BlochSimulators: _to_gpu_array, _is_gpu_array,
    initialize_states, initial_conditions!, regrowth!, dephasing!,
    diffusion_decay_matrix, sample_transverse!, sample_Ω!,
    simulate_magnetization, simulate_magnetization!, _allocate_array_on_resource,
    _signal_per_coil!, _simulate!, _check_gpu, _mt_add_equilibrium

# Every NVIDIA GPU has 32 threads per warp. This is deliberately a literal
# rather than `CUDA.warpsize(device())`: a constant is computed when the
# package is precompiled, and on a cluster that usually happens on a login
# node without a GPU.
const WARPSIZE = Int32(32)
const THREADS_PER_BLOCK = 64

_to_gpu_array(x::AbstractArray) = CuArray(x)
_is_gpu_array(::CuArray) = true

include("epg.jl")
include("mt.jl")
include("simulate.jl")
include("check_sequence.jl")

end # module
