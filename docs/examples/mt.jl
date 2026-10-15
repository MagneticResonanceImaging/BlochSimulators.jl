# # Magnetization transfer

using Pkg;
Pkg.activate("docs");

# In this example we demonstrate how to simulate magnetization transfer (MT) effects
# with the two-pool (free + bound pool) models in BlochSimulators.jl. Two sequences
# implement the same spoiled gradient-echo scenario:
#
# - `MTFISP2D`: extended phase graph (EPG) model, with exact gradient spoiling.
# - `MTSPGR2D`: isochromat model, which emulates gradient spoiling by averaging over an
#   ensemble of `Niso` isochromats with different phase increments per TR.
#
# For both, every RF pulse rotates the free pool and partially saturates the bound
# pool, and the two pools exchange longitudinal magnetization in between pulses.

using BlochSimulators
using StructArrays
using PythonPlot

# ## Tissue properties

# Two-pool tissue properties hold, next to `T₁` and `T₂` of the free pool, the bound
# pool's `T₁ᵇ` and `T₂ᵇ`, the bound pool fraction `f` and the exchange rate `k` (s⁻¹)
# from the free to the bound pool. All times are in seconds. The values below are
# white-matter-like (the MT test case of EPG-X, Malik et al., MRM 2018).
T₁, T₂ = 0.779, 0.045
T₁ᵇ, T₂ᵇ = 0.779, 12e-6
f, k = 0.117, 4.3

wm = T₁T₂MT(T₁, T₂, T₁ᵇ, T₂ᵇ, f, k)

# Setting `f = 0` removes the bound pool, which gives the single-pool reference.
no_mt = T₁T₂MT(T₁, T₂, T₁ᵇ, T₂ᵇ, 0.0, 0.0)

# Use `T₁T₂B₁MT` instead to include B₁ inhomogeneity.

# ## Sequence

# A train of 10° pulses with quadratic RF spoiling. `abs.(RF_train)` are the flip angles in
# degrees and `angle.(RF_train)` the RF phases in radians.
nTR = 300
TR = 0.005
α = 10.0
ϕ = [n * (n - 1) / 2 * deg2rad(117.0) for n = 1:nTR]
RF_train = α .* cis.(ϕ)

# The saturation of the bound pool depends on the energy of each pulse, so the sequences
# also need the duration `τ_RF` of an RF pulse (s), a pulse-shape factor (`1.0` for a
# rectangular pulse), the RF frequency offset `Δ` (Hz) and the lineshape of the bound
# pool (`Gaussian()`, `Lorentzian()` or `SuperLorentzian()`).
τ_RF = 0.0005
pulse_shape_factor = 1.0
Δ = 0.0
lineshape = SuperLorentzian()

# The EPG version keeps track of `max_state` configuration states (a multiple of 32):
max_state = 64
epg = MTFISP2D(RF_train, TR, τ_RF, pulse_shape_factor, Δ, lineshape, Val(max_state))

# The isochromat version takes the number of isochromats `Niso` in the spoiling ensemble
# instead. This is unrelated to `max_state`: it sets how well the ensemble approximates
# ideal spoiling, while `max_state` truncates the EPG states.
Niso = 64
iso = MTSPGR2D(RF_train, TR, τ_RF, pulse_shape_factor, Δ, lineshape, Niso)

# ## Single voxel

# The magnetization is sampled right after each RF pulse. Demodulate the RF phase to
# get the signal.
demodulate(m) = vec(m) .* cis.(-ϕ)

s_epg = demodulate(simulate_magnetization(epg, wm))
s_iso = demodulate(simulate_magnetization(iso, wm))
s_no_mt = demodulate(simulate_magnetization(epg, no_mt))

# Saturating the bound pool lowers the free pool's steady-state signal:
println("Steady-state |signal|, with MT:    ", abs(s_epg[end]))
println("Steady-state |signal|, without MT: ", abs(s_no_mt[end]))

# and the isochromat model approaches the EPG model as `Niso` grows. Only the
# magnitudes are compared: the EPG and isochromat operators use opposite RF phase
# conventions, so the phases of the two signals differ.
println("Max |signal| difference EPG vs isochromat: ", maximum(abs.(abs.(s_epg) .- abs.(s_iso))))

figure()
plot(abs.(s_no_mt), label="MTFISP2D, f = 0 (no MT)")
plot(abs.(s_epg), label="MTFISP2D, f = $f")
plot(abs.(s_iso), "--", label="MTSPGR2D, f = $f")
xlabel("TR index")
ylabel("|signal| [a.u.]")
legend()
title("Spoiled gradient echo with magnetization transfer")

# ## Multiple voxels

# As for the single-pool models, simulating many voxels at once (for example to
# generate a dictionary) takes a `StructArray` of tissue properties. Here the bound
# pool fraction is varied:
fs = 0.0:0.05:0.25
parameters = StructArray(T₁T₂MT.(T₁, T₂, T₁ᵇ, T₂ᵇ, fs, k))
dictionary = simulate_magnetization(epg, parameters)

figure()
for (i, fᵢ) in enumerate(fs)
    plot(abs.(demodulate(dictionary[:, i])), label="f = $fᵢ")
end
xlabel("TR index")
ylabel("|signal| [a.u.]")
legend()
title("Effect of the bound pool fraction")

# The simulations also run multithreaded (`CPUThreads()`) or, after `using CUDA`, on
# the GPU, with `simulate_magnetization(gpu(f32(epg)), gpu(f32(parameters)))`.
