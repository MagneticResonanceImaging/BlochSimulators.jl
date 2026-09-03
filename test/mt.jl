@testset "Test magnetization transfer (MT) lineshape and exchange_propagator functions" begin

    # Gaussian and Lorentzian lineshapes are even in Δ and maximal at Δ=0
    T₂ᵇ = 12e-6
    @test BlochSimulators.lineshape(0.0, T₂ᵇ, Gaussian()) ≈ BlochSimulators.lineshape(-0.0, T₂ᵇ, Gaussian())
    @test BlochSimulators.lineshape(500.0, T₂ᵇ, Gaussian()) ≈ BlochSimulators.lineshape(-500.0, T₂ᵇ, Gaussian())
    @test BlochSimulators.lineshape(0.0, T₂ᵇ, Gaussian()) > BlochSimulators.lineshape(500.0, T₂ᵇ, Gaussian())

    @test BlochSimulators.lineshape(500.0, T₂ᵇ, Lorentzian()) ≈ BlochSimulators.lineshape(-500.0, T₂ᵇ, Lorentzian())
    @test BlochSimulators.lineshape(0.0, T₂ᵇ, Lorentzian()) > BlochSimulators.lineshape(500.0, T₂ᵇ, Lorentzian())

    # Super-Lorentzian: even in Δ, continuous across the interpolation boundary (Δ=1.5kHz)
    @test BlochSimulators.lineshape(2000.0, T₂ᵇ, SuperLorentzian()) ≈ BlochSimulators.lineshape(-2000.0, T₂ᵇ, SuperLorentzian())
    g_below = BlochSimulators.lineshape(1499.0, T₂ᵇ, SuperLorentzian())
    g_above = BlochSimulators.lineshape(1501.0, T₂ᵇ, SuperLorentzian())
    @test g_below ≈ g_above rtol = 1e-2

    # saturation_exponent: zero flip angle or zero pulse energy gives zero saturation
    @test BlochSimulators.saturation_exponent(0.0, 1e-3, 1.0, 0.0, T₂ᵇ, SuperLorentzian()) == 0.0
    @test BlochSimulators.saturation_exponent(90.0, 1e-3, 1.0, 0.0, T₂ᵇ, SuperLorentzian()) > 0.0

    # exchange_propagator: with no exchange (k=0,f=0), pool a reduces to plain T₁
    # decay/regrowth and is fully decoupled from pool b
    Δt, T₁ᵃ, T₁ᵇ = 0.5, 0.8, 1.2
    Λ, c = BlochSimulators.exchange_propagator(Δt, T₁ᵃ, T₁ᵇ, 0.0, 0.0)
    E₁ᵃ = exp(-Δt / T₁ᵃ)
    @test Λ[1, 1] ≈ E₁ᵃ
    @test Λ[1, 2] ≈ 0.0 atol = 1e-12
    @test Λ[2, 1] ≈ 0.0 atol = 1e-12
    @test c[1] ≈ 1 - E₁ᵃ
    @test c[2] ≈ 0.0 atol = 1e-12

    # with exchange, (1-f,f) (the two-pool equilibrium) is a fixed point of the propagator...
    f, k = 0.15, 2.0
    Λ2, c2 = BlochSimulators.exchange_propagator(Δt, T₁ᵃ, T₁ᵇ, k, f)
    Meq = SVector(1 - f, f)
    @test Λ2 * Meq + c2 ≈ Meq

    # ...and repeatedly applying the propagator from an arbitrary starting point
    # converges to it
    z = SVector(0.0, 0.0)
    for _ = 1:10_000
        z = Λ2 * z + c2
    end
    @test z ≈ Meq atol = 1e-6
end

@testset "Test operator functions for two-pool isochromat model" begin

    # initial_conditions: free pool starts at (0,0,1), bound pool at its equilibrium f
    p = T₁T₂MT(1.0, 0.1, 1.0, 1e-5, 0.2, 1.0)
    m0 = BlochSimulators.initial_conditions(TwoPoolIsochromat(0.0, 0.0, 0.0, 0.0), p)
    @test m0 == TwoPoolIsochromat(0.0, 0.0, 1.0, 0.2)

    # saturate: Wτ=0 is a no-op, large Wτ drives the bound pool to zero, free pool
    # is never affected
    m = TwoPoolIsochromat(0.3, -0.2, 0.5, 0.4)
    @test BlochSimulators.saturate(m, 0.0) == m
    m_sat = BlochSimulators.saturate(m, 50.0)
    @test m_sat.x == m.x && m_sat.y == m.y && m_sat.z == m.z
    @test abs(m_sat.zᵇ) < 1e-10

    # rotate: reduces exactly to the single-pool `rotate` on (x,y,z); zᵇ is untouched
    p2 = T₁T₂B₁MT(1.0, 0.1, 1.0, 1.0, 1e-5, 0.1, 1.0)
    γΔtRF = π / 2 + 0.0im
    γΔtGR, z, Δt = 0.0, 0.0, 0.0
    m = TwoPoolIsochromat(0.0, 0.0, 1.0, 0.42)
    m1 = BlochSimulators.Isochromat(0.0, 0.0, 1.0)
    mr = BlochSimulators.rotate(m, γΔtRF, γΔtGR, z, Δt, p2)
    mr1 = BlochSimulators.rotate(m1, γΔtRF, γΔtGR, z, Δt, p2)
    @test mr.x == mr1.x && mr.y == mr1.y && mr.z == mr1.z
    @test mr.zᵇ == m.zᵇ

    # exchange_relax: T₂ decay on (x,y), (z,zᵇ) follow the exchange propagator exactly
    E₂ᵃ = 0.7
    m = TwoPoolIsochromat(1.0, -0.5, 0.6, 0.2)
    Λ, c = BlochSimulators.exchange_propagator(0.02, 0.8, 1.1, 3.0, 0.15)
    mr2 = BlochSimulators.exchange_relax(m, E₂ᵃ, Λ, c)
    @test mr2.x ≈ m.x * E₂ᵃ
    @test mr2.y ≈ m.y * E₂ᵃ
    zᵃzᵇ = Λ * SVector(m.z, m.zᵇ) + c
    @test mr2.z ≈ zᵃzᵇ[1]
    @test mr2.zᵇ ≈ zᵃzᵇ[2]

    # invert: free pool inverted exactly like the single-pool `invert`, bound pool
    # saturated (not coherently rotated) using its own pulse parameters
    m = TwoPoolIsochromat(0.0, 0.0, 1.0, 0.3)
    m1 = BlochSimulators.Isochromat(0.0, 0.0, 1.0)
    mi = BlochSimulators.invert(m, p2, 1e-3, 1.0, 0.0, SuperLorentzian())
    mi1 = BlochSimulators.invert(m1, p2)
    @test mi.x == mi1.x && mi.y == mi1.y && mi.z == mi1.z
    @test mi.zᵇ < m.zᵇ # saturated (partially or fully) towards zero

    # adiabatic invert: free pool sign-flipped, bound pool untouched (B₁-insensitive,
    # no saturation parameters supplied)
    mia = BlochSimulators.invert(m)
    @test mia == TwoPoolIsochromat(0.0, 0.0, -1.0, 0.3)
end

@testset "Test operator functions for two-pool EPG model" begin

    Ns = 32
    f = 0.15

    # mt_initial_conditions!: free pool exactly as `initial_conditions!`, bound pool
    # starts at its equilibrium f at order 0 only
    Ω = zeros(ComplexF64, 3, Ns) |> ConfigurationStates
    Zᵇ = zeros(SVector{Ns,ComplexF64})
    Zᵇ = BlochSimulators.mt_initial_conditions!(Ω, Zᵇ, f)
    @test Ω[3, 1] == 1.0 + 0.0im
    @test all(Ω[1:2, :] .== 0)
    @test Zᵇ[1] == f + 0.0im
    @test all(Zᵇ[2:end] .== 0)

    # mt_saturate: order-independent scaling, matching `saturation_exponent`
    @test BlochSimulators.mt_saturate(Zᵇ, 0.0) == Zᵇ
    Zᵇsat = BlochSimulators.mt_saturate(Zᵇ, 50.0)
    @test abs(Zᵇsat[1]) < 1e-10

    # mt_exchange_relax!: order 0 matches the plain two-state `exchange_propagator`
    # exactly; higher orders relax without the equilibrium-regrowth term (which only
    # applies to the unmodulated, order-0 component)
    Δt, T₁ᵃ, T₁ᵇ, k = 0.02, 0.8, 1.1, 3.0
    Λ, c = BlochSimulators.exchange_propagator(Δt, T₁ᵃ, T₁ᵇ, k, f)
    E₂ᵃ = 0.9

    Ω = zeros(ComplexF64, 3, Ns) |> ConfigurationStates
    Ω[1, 1], Ω[2, 1], Ω[3, 1] = 0.3, 0.3, 0.6
    Ω[3, 2] = 0.25 # a nonzero higher-order Zᵃ state
    Zᵇvec = zeros(ComplexF64, Ns)
    Zᵇvec[1] = 0.4
    Zᵇ = SVector{Ns}(Zᵇvec)

    Zᵇnew = BlochSimulators.mt_exchange_relax!(Ω, Zᵇ, E₂ᵃ, Λ, c)

    @test Ω[1, 1] ≈ 0.3 * E₂ᵃ
    @test Ω[2, 1] ≈ 0.3 * E₂ᵃ

    z0 = Λ * SVector(0.6 + 0im, 0.4 + 0im) + c
    @test Ω[3, 1] ≈ z0[1]
    @test Zᵇnew[1] ≈ z0[2]

    z1 = Λ * SVector(0.25 + 0im, 0.0 + 0im) # no equilibrium injection at order > 0
    @test Ω[3, 2] ≈ z1[1]
    @test Zᵇnew[2] ≈ z1[2]

    # excite!/dephasing!/spoil! are completely unmodified by the two-pool extension
    # (Ω stays exactly 3×Ns; only the separate Zᵇ vector is new), so the existing
    # single-pool operator tests already cover them.
end

@testset "Test two-pool MT sequences on different computational resources" begin

    nTR = 200
    RF_train = complex.(ones(nTR) .* 15.0)
    nvoxels = 20
    parameters = [T₁T₂B₁MT(1.0, 0.08 + 0.01v, 0.9, 1.0, 1e-5, 0.1 + 0.01v, 2.0) for v = 1:nvoxels] |> StructArray

    sequences = (
        MTSPGR2D(RF_train, 0.010, 0.001, 1.0, 0.0, SuperLorentzian(), 32),
        MTFISP2D(RF_train, 0.010, 0.001, 1.0, 0.0, SuperLorentzian(), Val(32)),
    )

    for sequence in sequences
        m_cpu1 = simulate_magnetization(CPU1(), sequence, parameters)
        m_cputhreads = simulate_magnetization(CPUThreads(), sequence, parameters)
        @test m_cpu1 ≈ m_cputhreads

        if CUDA.functional()
            m_cpu_f32 = simulate_magnetization(CPU1(), f32(sequence), f32(parameters))
            m_gpu = simulate_magnetization(CUDALibs(), gpu(f32(sequence)), gpu(f32(parameters))) |> collect
            @test m_cpu_f32 ≈ m_gpu
        end
    end
end

@testset "Test two-pool isochromat sequence converges to the two-pool EPG sequence" begin

    # MTSPGR2D emulates ideal spoiling with a finite ensemble of Niso isochromats, while
    # MTFISP2D spoils exactly (analytically, via `dephasing!`). For a constant-phase RF
    # train, increasing Niso should make MTSPGR2D converge to MTFISP2D.
    nTR = 30
    RF_train = complex.(ones(nTR) .* 20.0)
    p = T₁T₂MT(1.0, 0.1, 1.0, 10e-6, 0.117, 4.3)

    seq_epg = MTFISP2D(RF_train, 0.010, 0.001, 1.0, 0.0, SuperLorentzian(), Val(32))
    m_epg = simulate_magnetization(seq_epg, p)

    err_prev = Inf
    for Niso in (8, 32, 128)
        seq_iso = MTSPGR2D(RF_train, 0.010, 0.001, 1.0, 0.0, SuperLorentzian(), Niso)
        m_iso = simulate_magnetization(seq_iso, p)
        err = maximum(abs.(m_iso .- m_epg))
        @test err < err_prev || err < 1e-8
        err_prev = err
    end
    @test err_prev < 1e-6
end

@testset "Test MT sequences reduce to the known analytical single-pool RF-spoiled steady-state" begin

    # With f=0 (no bound pool) and standard quadratic RF-spoiling phase cycling
    # (Zur et al.), a gradient- and RF-spoiled sequence's steady-state signal is well
    # approximated by the classic Ernst-angle spoiled-GRE formula. This is an
    # independent (textbook), external check on top of the EPG-X validation script.
    T₁, T₂, TR, θ = 1.0, 0.1, 0.010, deg2rad(15.0)
    E₁ = exp(-TR / T₁)
    Mss_ernst = sin(θ) * (1 - E₁) / (1 - E₁ * cos(θ))

    nTR = 1500
    Φ₀ = deg2rad(117.0)
    ϕ = [p * (p - 1) / 2 * Φ₀ for p = 1:nTR]
    RF_train = complex.(rad2deg(θ) .* cos.(ϕ), rad2deg(θ) .* sin.(ϕ))

    p_mt = T₁T₂MT(T₁, T₂, 1.0, 10e-6, 0.0, 0.0) # f=0: no bound pool

    seq_epg = MTFISP2D(RF_train, TR, 0.001, 1.0, 0.0, SuperLorentzian(), Val(64))
    m_epg = simulate_magnetization(seq_epg, p_mt)
    sig_epg = m_epg[:] .* exp.(-im .* ϕ) # demodulate RF phase, as in EPG-X

    seq_iso = MTSPGR2D(RF_train, TR, 0.001, 1.0, 0.0, SuperLorentzian(), 64)
    m_iso = simulate_magnetization(seq_iso, p_mt)
    sig_iso = m_iso[:] .* exp.(-im .* ϕ)

    @test abs(sig_epg[end]) ≈ Mss_ernst rtol = 0.05
    @test abs(sig_iso[end]) ≈ Mss_ernst rtol = 0.05
    @test abs(sig_epg[end]) ≈ abs(sig_iso[end]) rtol = 1e-3
end
