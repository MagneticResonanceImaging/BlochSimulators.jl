```@meta
CurrentModule = BlochSimulators
```

# Validation

Besides unit tests of the individual operators, the test suite compares
complete simulations with closed-form solutions from the MR literature
(`test/validation.jl`). Each sequence is run long enough to reach its steady
state, and the magnitude of the simulated transverse magnetization at the echo
time is compared with the closed-form expression. The forward-sensitivity
derivatives ([`simulate_derivatives_forward_sensitivity`](@ref)) are compared
with the derivatives of the same expressions w.r.t. $T_1$, $T_2$ and $B_1$
(which scales the flip angle $\alpha$), so that they are checked independently
of the simulator itself.

Below, $E_1 = e^{-TR/T_1}$ and $E_2 = e^{-TR/T_2}$, and the expressions give
the transverse magnetization right after the RF pulse, per unit of $M_0$. The
simulations are compared at the echo time, i.e. after an additional factor
$e^{-TE/T_2}$.

| Sequence | Simulator | Model | Agreement |
|:---------|:----------|:------|:----------|
| Spoiled gradient echo, ideal spoiling | a minimal test sequence built from the EPG operators | EPG | ~1e-15 |
| Inversion recovery, single excitation | [`FISP2D`](@ref) | EPG | ~1e-15 |
| Unbalanced SSFP (FISP) | [`FISP3D`](@ref) | EPG | ~1e-14, see below |
| Balanced SSFP, with off-resonance | [`pSSFP3D`](@ref) | isochromat | ~1e-14 |
| Balanced SSFP, from waveforms | [`Generic3D`](@ref) | isochromat | ~1e-8, see below |

The derivatives agree to within ~1e-7 (relative), which is the accuracy of the
finite differences of the closed-form expressions they are compared with.

## Spoiled gradient echo

With ideal spoiling of the transverse magnetization at the end of every TR,
the steady state is given by the Ernst formula (Ernst & Anderson, 1966):

$M_{xy} = \sin\alpha \frac{1 - E_1}{1 - E_1 \cos\alpha}.$

## Inversion recovery

A single excitation a time $TI$ after an ideal inversion gives

$M_{xy} = \sin\alpha \left(1 - 2 e^{-TI/T_1}\right).$

## Unbalanced SSFP (FISP)

For a gradient-spoiled sequence without RF spoiling, in which the dephasing
gradient dephases the magnetization completely within a voxel, the steady-state
FID signal is (Hänicke & Vogel, 2003)

$M_{xy} = \tan\frac{\alpha}{2} \left(1 - \frac{(E_1 - \cos\alpha)(1 - E_2^2)}{\sqrt{p^2 - q^2}}\right),$

with $p = 1 - E_1\cos\alpha - E_2^2(E_1 - \cos\alpha)$ and $q = E_2(1 - E_1)(1 + \cos\alpha)$.

This is the limit of an infinite number of configuration states. With
`max_state` states, the EPG simulation differs from it by a relative amount of
the order of $E_2^{\text{max\_state}}$: for example ~3e-7 with 128 states at
$TR = 10$ ms and $T_2 = 80$ ms. The test uses short enough $T_2$s (relative to
TR) for this to be negligible.

## Balanced SSFP

For a balanced sequence with alternating RF phase and off-resonance precession
$\theta = 2\pi \, \Delta B_0 \, TR$ per TR (Freeman & Hill, 1971),

$M_{xy} = \frac{(1 - E_1) \sin\alpha \sqrt{1 - 2E_2\cos\theta' + E_2^2}}{(1 - E_1\cos\alpha)(1 - E_2\cos\theta') - (E_1 - \cos\alpha)(E_2 - \cos\theta')E_2},$

with $\theta' = \theta + \pi$ to account for the alternating RF phase. The echo
time is $TR/2$. The test uses [`pSSFP3D`](@ref) with a single, instantaneous RF
sample, and the same sequence written out as RF and gradient waveforms for
[`Generic3D`](@ref). The latter needs a finite RF duration; with a 1 ns block
pulse, relaxation during the pulse causes the ~1e-8 difference.

## References

- Ernst RR, Anderson WA. Application of Fourier transform spectroscopy to
  magnetic resonance. Rev Sci Instrum 1966;37:93–102.
- Freeman R, Hill HDW. Phase and intensity anomalies in Fourier transform NMR.
  J Magn Reson 1971;4:366–383.
- Hänicke W, Vogel HU. An analytical solution for the SSFP signal in MRI.
  Magn Reson Med 2003;49:771–775.
