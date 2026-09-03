#!/usr/bin/env julia
#
# Validate BlochSimulators' two-pool magnetization transfer (MT) implementation
# (`MTSPGR2D`, the isochromat model; `MTFISP2D`, the EPG model) against Shaihan Malik's
# EPG-X toolbox (github.com/mriphysics/EPG-X, Malik et al., MRM 2018), which implements
# the same two-pool (free + bound pool) binary spin-bath model independently, in
# MATLAB. This is an external, independent-codebase check on top of the unit and
# CPU/GPU consistency tests in `test/runtests.jl`.
#
# The scenario below (TR, flip angle, RF-spoiling increment, T1/T2/f/k) is taken
# directly from EPG-X's own `test1_steady_state_GRE.m` MT test case, so the comparison
# uses parameters the EPG-X authors themselves chose to demonstrate their method.
#
# What gets compared:
#   1. BlochSimulators' MTFISP2D (EPG)         vs EPG-X's `EPGX_GRE_MT`   (EPG)
#   2. BlochSimulators' MTSPGR2D (isochromat)   vs EPG-X's `isochromat_GRE_MT`
#      (both isochromat implementations use an ensemble of `Niso` isochromats to
#      emulate spoiling; compared at matched Niso, and Niso is increased to show
#      convergence -- this mirrors the convergence check EPG-X's own paper/test1
#      script performs against its EPG result)
#   3. Both EPG results                        vs the closed-form analytic
#      steady-state formula `ssSPGR_MT` (an independent, third check)
#
# Complex signals are compared by magnitude only: BlochSimulators and EPG-X do not
# necessarily share the same global transverse-phase reference convention (e.g.
# EPG-X's F0 demodulation includes an extra `*1i` factor), and that convention choice
# carries no physical meaning -- the magnitude comparison is what actually tests the
# physics.
#
# Requirements: `git` and `matlab` (with `-batch` support) on PATH; internet access to
# clone EPG-X once (cached under a temp directory for subsequent runs).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BlochSimulators

### 1. Fetch EPG-X (cached; only cloned once) ################################

const EPGX_ROOT = joinpath(tempdir(), "BlochSimulators_EPGX_validation")
const EPGX_DIR = joinpath(EPGX_ROOT, "EPG-X")

if !isdir(EPGX_DIR)
    println("Cloning EPG-X into $EPGX_DIR ...")
    mkpath(EPGX_ROOT)
    run(`git clone --depth 1 https://github.com/mriphysics/EPG-X.git $EPGX_DIR`)
else
    println("Using cached EPG-X checkout at $EPGX_DIR")
end

matlab_bin = something(Sys.which("matlab"), isfile("/packages/bin/matlab") ? "/packages/bin/matlab" : nothing)
matlab_bin === nothing && error("Could not find a `matlab` executable on PATH.")

### 2. Scenario: EPG-X's test1_steady_state_GRE.m MT parameters ##############

TR_ms = 5.0
alpha_deg = 10.0
phi0_deg = 117.0          # quadratic RF-spoiling phase increment (Zur et al.)
T1a_ms, T1b_ms = 779.0, 779.0
T2a_ms = 45.0
T2b_s = 12e-6              # bound pool T₂ (seconds); EPG-X's own test hardcodes G
                            # directly instead of a T₂ᵇ, so both sides here compute G
                            # from this same T₂ᵇ via their own (independent) lineshape
                            # implementation -- this also validates the lineshape code.
f = 0.117
k_per_ms = 4.3e-3
B1_uT = 13.0                # only used, here, to construct a concrete (flip,τ_RF) pair
γ_rad_per_ms_per_uT = 267.5221e-3   # EPG-X's own γ constant (rad·ms⁻¹·μT⁻¹)
npulse = 200
Niso_list = (8, 32, 128)

τ_RF_ms = deg2rad(alpha_deg) / (γ_rad_per_ms_per_uT * B1_uT)
τ_RF_s = τ_RF_ms * 1e-3

# quadratic RF-spoiling phase cycling, matching EPG-X's `RF_phase_cycle(npulse,phi0)`
ϕ = [p * (p - 1) / 2 * deg2rad(phi0_deg) for p = 1:npulse]
RF_train = complex.(alpha_deg .* cos.(ϕ), alpha_deg .* sin.(ϕ))

p_mt = T₁T₂MT(T1a_ms / 1000, T2a_ms / 1000, T1b_ms / 1000, T2b_s, f, k_per_ms * 1000)

### 3. Run BlochSimulators #####################################################

println("Running BlochSimulators MTFISP2D (EPG)...")
seq_epg = MTFISP2D(RF_train, TR_ms / 1000, τ_RF_s, 1.0, 0.0, SuperLorentzian(), Val(64))
m_epg = simulate_magnetization(seq_epg, p_mt)
sig_epg = vec(m_epg) .* exp.(-im .* ϕ) # demodulate RF phase

println("Running BlochSimulators MTSPGR2D (isochromat) for Niso = $Niso_list ...")
sig_iso = Dict{Int,Vector{ComplexF64}}()
for Niso in Niso_list
    seq_iso = MTSPGR2D(RF_train, TR_ms / 1000, τ_RF_s, 1.0, 0.0, SuperLorentzian(), Niso)
    m_iso = simulate_magnetization(seq_iso, p_mt)
    sig_iso[Niso] = vec(m_iso) .* exp.(-im .* ϕ)
end

### 4. Run EPG-X in MATLAB #####################################################

workdir = mktempdir()
mscript = joinpath(workdir, "run_epgx_mt.m")

niso_list_matlab = "[" * join(Niso_list, " ") * "]"

open(mscript, "w") do io
    write(io, """
    addpath(genpath('$(EPGX_DIR)/lib'));
    addpath(genpath('$(EPGX_DIR)/EPGX-src'));

    TR = $(TR_ms);
    alpha = $(alpha_deg);
    phi0 = $(phi0_deg);
    T1_MT = [$(T1a_ms) $(T1b_ms)];
    T2_MT = $(T2a_ms);
    f_MT = $(f);
    k_MT = $(k_per_ms);
    T2b = $(T2b_s);
    b1sqrdtau = $(B1_uT)^2 * $(τ_RF_ms);
    npulse = $(npulse);
    Niso_list = $(niso_list_matlab);

    [ff, Garr] = SuperLorentzian(T2b);
    G = interp1(ff, Garr, 0);

    phi = RF_phase_cycle(npulse, phi0);
    theta = d2r(alpha) * ones(npulse, 1);
    b1sqrdtau_arr = b1sqrdtau * ones(npulse, 1);

    [smt, ~, ~] = EPGX_GRE_MT(theta, phi, b1sqrdtau_arr, TR, T1_MT, T2_MT, f_MT, k_MT, G);

    fid = fopen('$(joinpath(workdir, "smt.txt"))', 'w');
    fprintf(fid, '%.12e %.12e\\n', [real(smt(:))'; imag(smt(:))']);
    fclose(fid);

    for ii = 1:numel(Niso_list)
        s = isochromat_GRE_MT(theta, phi, b1sqrdtau_arr, TR, T1_MT, T2_MT, f_MT, k_MT, G, Niso_list(ii));
        fid = fopen(sprintf('$(joinpath(workdir, "siso_%d.txt"))', Niso_list(ii)), 'w');
        fprintf(fid, '%.12e %.12e\\n', [real(s(:))'; imag(s(:))']);
        fclose(fid);
    end

    Mss = ssSPGR_MT(d2r(alpha), b1sqrdtau, TR, T1_MT, f_MT, k_MT, G);
    fid = fopen('$(joinpath(workdir, "scalar.txt"))', 'w');
    fprintf(fid, '%.12e %.12e %.12e\\n', real(Mss), imag(Mss), G);
    fclose(fid);
    """)
end

println("Running EPG-X in MATLAB...")
run(`$matlab_bin -batch "run('$mscript')"`)

# --- read results back (plain text: one "re im" pair per line, no extra deps needed)
function read_complex_column(path)
    v = ComplexF64[]
    for line in readlines(path)
        isempty(strip(line)) && continue
        re_s, im_s = split(line)
        push!(v, parse(Float64, re_s) + 1im * parse(Float64, im_s))
    end
    return v
end

smt_epgx = read_complex_column(joinpath(workdir, "smt.txt"))
siso_epgx = Dict(Niso => read_complex_column(joinpath(workdir, "siso_$(Niso).txt")) for Niso in Niso_list)
Mss_re, Mss_im, G_used = parse.(Float64, split(readline(joinpath(workdir, "scalar.txt"))))
Mss_epgx = Mss_re + 1im * Mss_im

### 5. Compare and report ######################################################

relerr(a, b) = maximum(abs.(abs.(a) .- abs.(b))) / maximum(abs.(b))

println()
println("=" ^ 72)
println("EPG-X validation results  (lineshape G = $(G_used) μs, from T₂ᵇ = $(T2b_s*1e6) μs)")
println("=" ^ 72)

println(rpad("BlochSimulators MTFISP2D (EPG)  vs EPG-X EPGX_GRE_MT", 58), "max rel err (mag) = ", relerr(sig_epg, smt_epgx))

for Niso in Niso_list
    println(rpad("BlochSimulators MTSPGR2D (Niso=$Niso) vs EPG-X isochromat_GRE_MT (Niso=$Niso)", 58),
        "max rel err (mag) = ", relerr(sig_iso[Niso], siso_epgx[Niso]))
end

println(rpad("BlochSimulators MTSPGR2D convergence to MTFISP2D as Niso increases:", 58))
for Niso in Niso_list
    println("    Niso=$Niso  max rel err (mag) vs BlochSimulators MTFISP2D = ", relerr(sig_iso[Niso], sig_epg))
end

println()
println(rpad("Closed-form steady-state ssSPGR_MT |M|", 58), Mss_epgx |> abs)
println(rpad("BlochSimulators MTFISP2D  last-pulse |M|", 58), abs(sig_epg[end]))
println(rpad("EPG-X EPGX_GRE_MT        last-pulse |M|", 58), abs(smt_epgx[end]))

println("=" ^ 72)
