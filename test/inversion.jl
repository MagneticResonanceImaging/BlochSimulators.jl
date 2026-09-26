@testset "Test inversion models" begin

    five_point_central(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)

    # Example inversion pulse (dur 9.4944 ms, b1 13.4978 µT)
    pulse = hyperbolic_secant_pulse(9.4944e-3, 13.4978e-6)
    model = EffectiveAdiabaticInversion(pulse)
    bloch_Mz(T₁, T₂, B₁) = simulate_magnetization(CPU1(), pulse, StructVector([T₁T₂B₁(T₁, T₂, B₁)]))[1].z

    # 0.6 is a grid point (spacing 0.02): the B₁ derivative must be continuous there
    testpoints = [(1.0, 0.1, 0.87), (0.3, 0.02, 1.13), (2.0, 0.5, 0.6)]

    RF_train = complex.([30 + 25 * sin(2π * 4.0 * t / 24) for t = 1:24])
    fisp(inversion) = FISP3D(RF_train, 0.00558, 0.002745, Val(32), 0.016, 0.005, 2, true, true, 1, 0.0, inversion)

    @testset "fitted efficiency matches a Bloch simulation of the pulse" begin
        @test model.duration ≈ 9.4944e-3
        for T₁ in (0.3, 1.0, 3.0), T₂ in (0.005, 0.05, 0.25), B₁ in (0.51, 0.87, 1.0, 1.33)
            @test abs(inversion_efficiency(model, T₁T₂B₁(T₁, T₂, B₁)) - bloch_Mz(T₁, T₂, B₁)) < 6e-3
        end
        # tissue properties without B₁ are treated as B₁ = 1
        @test inversion_efficiency(model, T₁T₂(1.0, 0.1)) == inversion_efficiency(model, T₁T₂B₁(1.0, 0.1, 1.0))
        # outside the grid the end values are used
        @test inversion_efficiency(model, T₁T₂B₁(1.0, 0.1, 2.0)) ≈ inversion_efficiency(model, T₁T₂B₁(1.0, 0.1, 1.6))
        # the same pulse as a phase-modulated waveform in the carrier frame (as a Pulseq file
        # stores it) gives the same model
        phase = cumsum(-pulse.Δω .* pulse.Δt)
        phase_modulated = AdiabaticPulse(pulse.γΔtA .* cis.(phase), zero(pulse.Δω), pulse.Δt)
        model_pm = EffectiveAdiabaticInversion(phase_modulated)
        @test model_pm.η₀ ≈ model.η₀ atol = 1e-4
        @test model_pm.τ₁ ≈ model.τ₁ rtol = 1e-3
        @test model_pm.τ₂ ≈ model.τ₂ rtol = 1e-3
        # the pulse does not invert at very low B₁
        @test_throws ArgumentError EffectiveAdiabaticInversion(pulse; B₁=0.2:0.1:1.0)
        # rebuilding from the tabulated values (as when read back from a file) is lossless
        N = length(model.η₀)
        grid = range(model.B₁_start, step=model.ΔB₁, length=N)
        rebuilt = EffectiveAdiabaticInversion(model.duration, grid, collect(model.η₀), collect(model.τ₁), collect(model.τ₂))
        @test rebuilt === model
        @test_throws ArgumentError EffectiveAdiabaticInversion(model.duration, grid, collect(model.η₀)[1:end-1], collect(model.τ₁), collect(model.τ₂))
    end

    @testset "derivatives of the efficiency" begin
        for (T₁, T₂, B₁) in testpoints
            η, ∂η∂T₁, ∂η∂T₂, ∂η∂B₁ = BlochSimulators.inversion_efficiency_and_derivatives(model, T₁T₂B₁(T₁, T₂, B₁))
            η_at(T₁, T₂, B₁) = inversion_efficiency(model, T₁T₂B₁(T₁, T₂, B₁))
            @test η == η_at(T₁, T₂, B₁)
            @test ∂η∂T₁ ≈ five_point_central(t -> η_at(t, T₂, B₁), T₁, 1e-4 * T₁) rtol = 1e-6
            @test ∂η∂T₂ ≈ five_point_central(t -> η_at(T₁, t, B₁), T₂, 1e-4 * T₂) rtol = 1e-6
            # small step: at a grid point the second derivative jumps
            @test ∂η∂B₁ ≈ five_point_central(b -> η_at(T₁, T₂, b), B₁, 1e-6) rtol = 1e-6
        end
        # outside the grid η does not depend on B₁
        @test BlochSimulators.inversion_efficiency_and_derivatives(model, T₁T₂B₁(1.0, 0.1, 1.7))[4] == 0
    end

    @testset "FISP3D: an efficiency of exactly -1 is the ideal inversion" begin
        perfect = EffectiveAdiabaticInversion{Float64,3}(0.01, 0.5, 0.5, SVector(-1.0, -1.0, -1.0), zeros(SVector{3}), zeros(SVector{3}))
        parameters = StructVector([T₁T₂B₁(T₁, T₂, B₁) for (T₁, T₂, B₁) in testpoints])

        # the constructor without an inversion model keeps the ideal inversion
        @test FISP3D(RF_train, 0.00558, 0.002745, Val(32), 0.016, 0.005, 2, true, true, 1, 0.0).inversion_model === IdealInversion()

        m_ideal = simulate_magnetization(CPU1(), fisp(IdealInversion()), parameters)
        @test simulate_magnetization(CPU1(), fisp(perfect), parameters) ≈ m_ideal
        @test !(simulate_magnetization(CPU1(), fisp(model), parameters) ≈ m_ideal)

        # shortening the sequence keeps the inversion model
        @test fisp(model)[1:5].inversion_model === model
    end

    @testset "FISP3D forward sensitivity (CPU Float64 vs 5-point central finite differences)" begin
        sequence = fisp(model)
        signal(T₁, T₂, B₁) = vec(simulate_magnetization(CPU1(), sequence, StructVector([T₁T₂B₁(T₁, T₂, B₁)])))

        for (T₁, T₂, B₁) in testpoints
            m, ∂m = simulate_derivatives_forward_sensitivity((:T₁, :T₂, :B₁), sequence, StructVector([T₁T₂B₁(T₁, T₂, B₁)]))

            @test vec(m) ≈ signal(T₁, T₂, B₁)
            @test isapprox(vec(∂m.T₁), five_point_central(t -> signal(t, T₂, B₁), T₁, 1e-6 * T₁); rtol=1e-5, atol=1e-8)
            @test isapprox(vec(∂m.T₂), five_point_central(t -> signal(T₁, t, B₁), T₂, 1e-6 * T₂); rtol=1e-5, atol=1e-8)
            @test isapprox(vec(∂m.B₁), five_point_central(b -> signal(T₁, T₂, b), B₁, 1e-6); rtol=1e-5, atol=1e-8)
        end
    end

    @testset "precision and GPU" begin
        model_f32 = f32(model)
        @test model_f32 isa EffectiveAdiabaticInversion{Float32}
        @test model_f32.η₀ ≈ model.η₀

        sequence = fisp(model)
        @test gpu(f32(sequence)).inversion_model isa EffectiveAdiabaticInversion{Float32}

        if CUDA.functional()
            nvoxels = 12
            T₁ = collect(exp.(range(log(0.1), log(5.0), length=nvoxels)))
            T₂ = collect(exp.(range(log(0.01), log(2.0), length=nvoxels)))
            B₁ = collect(range(0.5, 1.5, length=nvoxels))
            mask = T₁ .> T₂
            T₁, T₂, B₁ = T₁[mask], T₂[mask], B₁[mask]
            parameters = @parameters T₁ T₂ B₁

            derivatives = (:T₁, :T₂, :B₁)
            m_cpu, ∂m_cpu = simulate_derivatives_forward_sensitivity(derivatives, sequence, parameters)

            sequence_gpu = gpu(f32(sequence))
            parameters_gpu = gpu(f32(parameters))
            m_gpu, ∂m_gpu = simulate_derivatives_forward_sensitivity(derivatives, sequence_gpu, parameters_gpu)

            relerr(a, b) = norm(a - b) / norm(b)
            @test relerr(Array(m_gpu), m_cpu) < 1e-3
            @test relerr(Array(∂m_gpu.T₁), ∂m_cpu.T₁) < 1e-3
            @test relerr(Array(∂m_gpu.T₂), ∂m_cpu.T₂) < 1e-3
            @test relerr(Array(∂m_gpu.B₁), ∂m_cpu.B₁) < 1e-3

            m_gpu_signal_only = simulate_magnetization(CUDALibs(), sequence_gpu, parameters_gpu)
            @test isapprox(Array(m_gpu), Array(m_gpu_signal_only); rtol=1e-4)
            @test relerr(Array(m_gpu_signal_only), simulate_magnetization(CPU1(), sequence, parameters)) < 1e-3
        end
    end
end
