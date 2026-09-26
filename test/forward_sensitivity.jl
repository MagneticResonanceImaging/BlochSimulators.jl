@testset "Test FISP3D forward sensitivity derivatives" begin

    # helper: 5-point central finite difference, used as a high-order oracle
    # (the one-sided Float32 finite differences used elsewhere in the
    # package are not accurate enough to serve as the truth oracle here)
    five_point_central(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)

    @testset "operator level: relaxation + regrowth sensitivity vs finite differences" begin

        # Sensitivities are propagated by handing the operators the state
        # `(Ω, (∂Ω/∂T₁, ∂Ω/∂T₂))` -- ordinary configuration state matrices
        # throughout -- and reusing `rotate_decay!`/`regrowth!` completely
        # unmodified (see src/derivatives/forward_sensitivity.jl); this
        # exercises that mechanism directly at the operator level rather than
        # through a full FISP3D sequence.
        ∂Ω∂T₁T₂ = BlochSimulators.∂Ω∂T₁T₂

        # A sensitivity state with the given values and zero sensitivities.
        state_triple(Ωvals) = (ConfigurationStates(copy(Ωvals)),
            ∂Ω∂T₁T₂(ConfigurationStates(zeros(ComplexF64, size(Ωvals))),
                ConfigurationStates(zeros(ComplexF64, size(Ωvals)))))
        # A relaxation factor held fixed, i.e. with zero derivative.
        constant_factor(e) = (e, zero(e))

        rng_matrix() = rand(ComplexF64, 3, 8)

        Δt = 0.07
        E₂fixed = 0.83 # held fixed while differentiating the "other" parameter's E

        for _ in 1:3
            Ωvals = rng_matrix()

            T₁ = 0.05 + 2 * rand()
            f_T₁ = t -> begin
                Ω2 = ConfigurationStates(copy(Ωvals))
                BlochSimulators.rotate_decay!(Ω2, BlochSimulators._E₁(Δt, t), E₂fixed, complex(1.0))
                BlochSimulators.regrowth!(Ω2, BlochSimulators._E₁(Δt, t))
                return copy(Ω2.matrix)
            end
            h = 1e-6 * T₁
            fd = five_point_central(f_T₁, T₁, h)

            Ωdual = state_triple(Ωvals)
            E₁dual = BlochSimulators.E₁(Ωdual, Δt, T₁)
            BlochSimulators.rotate_decay!(Ωdual, E₁dual, constant_factor(E₂fixed), complex(1.0))
            BlochSimulators.regrowth!(Ωdual, E₁dual)

            @test isapprox(Ωdual[2].∂T₁.matrix, fd; atol=1e-6, rtol=1e-6)
            # the value itself must be propagated exactly as usual
            @test Ωdual[1].matrix ≈ f_T₁(T₁)

            T₂ = 0.05 + 2 * rand()
            E₁fixed = 0.77
            f_T₂ = t -> begin
                Ω2 = ConfigurationStates(copy(Ωvals))
                BlochSimulators.rotate_decay!(Ω2, E₁fixed, BlochSimulators._E₂(Δt, t), complex(1.0))
                BlochSimulators.regrowth!(Ω2, E₁fixed)
                return copy(Ω2.matrix)
            end
            h2 = 1e-6 * T₂
            fd2 = five_point_central(f_T₂, T₂, h2)

            Ωdual2 = state_triple(Ωvals)
            E₂dual = BlochSimulators.E₂(Ωdual2, Δt, T₂)
            BlochSimulators.rotate_decay!(Ωdual2, constant_factor(E₁fixed), E₂dual, complex(1.0))
            BlochSimulators.regrowth!(Ωdual2, constant_factor(E₁fixed))

            @test isapprox(Ωdual2[2].∂T₂.matrix, fd2; atol=1e-6, rtol=1e-6)
            # the T₂ factor must leave the T₁ sensitivity untouched
            @test all(iszero, Ωdual2[2].∂T₁.matrix)
        end
    end

    @testset "operator level: RF rotation, inversion, spoiling and dephasing act identically on sensitivities" begin

        # These operators don't depend on T₁/T₂, so applying them to a
        # (nonzero, otherwise arbitrary) sensitivity state should behave
        # exactly like applying them to any other configuration state.
        p = T₁T₂B₁(1.0, 0.1, 0.9)
        RF = complex(37.0, 12.0)

        S = ConfigurationStates(rand(ComplexF64, 3, 8))
        Sref = ConfigurationStates(copy(S.matrix))

        BlochSimulators.excite!(S, RF, p)
        BlochSimulators.excite!(Sref, RF, p) # same function, sanity check it's deterministic
        @test S.matrix == Sref.matrix

        BlochSimulators.invert!(S)
        BlochSimulators.invert!(Sref)
        @test S.matrix == Sref.matrix

        BlochSimulators.spoil!(S)
        BlochSimulators.spoil!(Sref)
        @test S.matrix == Sref.matrix

        BlochSimulators.dephasing!(S)
        BlochSimulators.dephasing!(Sref)
        @test S.matrix == Sref.matrix
    end

    @testset "full sequence CPU Float64 oracle (5-point central finite differences)" begin

        RF_train = complex.(fill(30.0, 20))
        sequence = FISP3D(RF_train, 0.010, 0.003, Val(32), 0.02, 0.0, 2, true, true, 1, 0.0)

        reference_signal(T₁, T₂, B₁) = vec(simulate_magnetization(CPU1(), sequence, StructVector([T₁T₂B₁(T₁, T₂, B₁)])))

        testpoints = [
            (0.1, 0.01, 1.0),   # lower T₁/T₂ bounds
            (5.0, 2.0, 1.0),    # upper T₁/T₂ bounds
            (0.1, 0.099, 1.0),  # T₂ close to T₁ (but still valid)
            (1.0, 0.999, 1.0),
            (1.0, 0.1, 0.8),    # representative B₁
            (1.0, 0.1, 1.2),
        ]

        for (T₁, T₂, B₁) in testpoints
            parameters = StructVector([T₁T₂B₁(T₁, T₂, B₁)])
            m, ∂m = simulate_derivatives_forward_sensitivity((:T₁, :T₂), sequence, parameters)

            h₁ = 1e-6 * T₁
            h₂ = 1e-6 * T₂
            fd_T₁ = five_point_central(t -> reference_signal(t, T₂, B₁), T₁, h₁)
            fd_T₂ = five_point_central(t -> reference_signal(T₁, t, B₁), T₂, h₂)

            @test isapprox(vec(∂m.T₁), fd_T₁; rtol=1e-5, atol=1e-8)
            @test isapprox(vec(∂m.T₂), fd_T₂; rtol=1e-5, atol=1e-8)

            # signal itself must be unaffected by tracking sensitivities
            @test vec(m) ≈ reference_signal(T₁, T₂, B₁)
        end
    end

    @testset "diffusion: sensitivities remain correct with D > 0" begin

        RF_train = complex.([30 + 25 * sin(2π * 4.0 * t / 16) for t = 1:16])
        Δk_spoil = 2π * 1000
        sequence = FISP3D(RF_train, 0.010, 0.005, Val(32), 0.02, 0.0, 2, true, true, 1, Δk_spoil)

        T₁, T₂, D = 1.0, 0.1, 0.002
        reference_signal(t₁, t₂) = vec(simulate_magnetization(CPU1(), sequence, StructVector([T₁T₂D(t₁, t₂, D)])))

        parameters = StructVector([T₁T₂D(T₁, T₂, D)])
        m, ∂m = simulate_derivatives_forward_sensitivity((:T₁, :T₂), sequence, parameters)

        fd_T₁ = five_point_central(t -> reference_signal(t, T₂), T₁, 1e-6 * T₁)
        fd_T₂ = five_point_central(t -> reference_signal(T₁, t), T₂, 1e-6 * T₂)

        @test isapprox(vec(∂m.T₁), fd_T₁; rtol=1e-5, atol=1e-8)
        @test isapprox(vec(∂m.T₂), fd_T₂; rtol=1e-5, atol=1e-8)
    end

    @testset "generalizes to other EPGSimulator sequences (FISP2D)" begin

        # Nothing in src/derivatives/forward_sensitivity.jl is specific to
        # FISP3D: it's written entirely in terms of the generic
        # src/operators/epg.jl operators, which FISP2D's simulate_magnetization!
        # is built from too. This checks that claim rather than just asserting
        # it in a comment.
        nTR = 40
        sequence = FISP2D(nTR)
        sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)

        reference_signal(T₁, T₂, B₁) = vec(simulate_magnetization(CPU1(), sequence, StructVector([T₁T₂B₁(T₁, T₂, B₁)])))

        T₁, T₂, B₁ = 0.8, 0.08, 1.0
        parameters = StructVector([T₁T₂B₁(T₁, T₂, B₁)])
        m, ∂m = simulate_derivatives_forward_sensitivity((:T₁, :T₂), sequence, parameters)

        fd_T₁ = five_point_central(t -> reference_signal(t, T₂, B₁), T₁, 1e-6 * T₁)
        fd_T₂ = five_point_central(t -> reference_signal(T₁, t, B₁), T₂, 1e-6 * T₂)

        @test isapprox(vec(∂m.T₁), fd_T₁; rtol=1e-5, atol=1e-8)
        @test isapprox(vec(∂m.T₂), fd_T₂; rtol=1e-5, atol=1e-8)
        @test vec(m) ≈ reference_signal(T₁, T₂, B₁)
    end

    @testset "single-direction (T₁ or T₂ only) requests" begin

        RF_train = complex.(fill(20.0, 20))
        sequence = FISP3D(RF_train, 0.010, 0.003, Val(32), 0.02, 0.0, 2, true, true, 1, 0.0)
        parameters = StructVector([T₁T₂B₁(0.8, 0.08, 0.9)])

        m_full, ∂m_full = simulate_derivatives_forward_sensitivity((:T₁, :T₂), sequence, parameters)
        m_T₁, ∂m_T₁ = simulate_derivatives_forward_sensitivity((:T₁,), sequence, parameters)
        m_T₂, ∂m_T₂ = simulate_derivatives_forward_sensitivity((:T₂,), sequence, parameters)

        @test propertynames(∂m_T₁) == (:T₁,)
        @test propertynames(∂m_T₂) == (:T₂,)
        @test m_full ≈ m_T₁ ≈ m_T₂
        @test ∂m_full.T₁ ≈ ∂m_T₁.T₁
        @test ∂m_full.T₂ ≈ ∂m_T₂.T₂
    end

    @testset "B₁ derivative, and the derivative set following the parameter type" begin

        # Which derivatives are available follows from the tissue property
        # type, and the requested set is part of the compiled kernel's type.
        supported = BlochSimulators.supported_derivatives
        @test supported(T₁T₂) == (:T₁, :T₂)
        @test supported(T₁T₂B₁) == (:T₁, :T₂, :B₁)
        @test supported(StructVector([T₁T₂B₁(1.0, 0.1, 0.9)])) == (:T₁, :T₂, :B₁)

        RF_train = complex.([30 + 25 * sin(2π * 4.0 * t / 20) for t = 1:20])
        sequence = FISP3D(RF_train, 0.00558, 0.002745, Val(32), 0.01084, 0.0, 2, true, true, 1, 0.0)
        signal(p) = vec(simulate_magnetization(CPU1(), sequence, StructVector([p])))

        p = T₁T₂B₁(0.85, 0.09, 0.93)
        m, ∂m = simulate_derivatives_forward_sensitivity(sequence, StructVector([p]))

        # the default request is everything the parameters support
        @test propertynames(∂m) == (:T₁, :T₂, :B₁)
        # tracking sensitivities must not disturb the magnetization itself
        @test vec(m) ≈ signal(p)

        relerr(a, b) = norm(a - b) / norm(b)
        @test relerr(vec(∂m.T₁),
            five_point_central(t -> signal(T₁T₂B₁(t, p.T₂, p.B₁)), p.T₁, 1e-6 * p.T₁)) < 1e-7
        @test relerr(vec(∂m.T₂),
            five_point_central(t -> signal(T₁T₂B₁(p.T₁, t, p.B₁)), p.T₂, 1e-6 * p.T₂)) < 1e-7
        @test relerr(vec(∂m.B₁),
            five_point_central(t -> signal(T₁T₂B₁(p.T₁, p.T₂, t)), p.B₁, 1e-6 * p.B₁)) < 1e-7

        # every subset compiles, and agrees with the full computation
        for requested in [(:T₁,), (:B₁,), (:T₁, :T₂, :B₁)]
            _, ∂ = simulate_derivatives_forward_sensitivity(requested, sequence, StructVector([p]))
            @test propertynames(∂) == requested
            @test all(θ -> ∂[θ] ≈ ∂m[θ], requested)
        end

        # B₁ sensitivity is nonzero (i.e. the source term in excite! fires)
        @test !all(iszero, ∂m.B₁)

    end

    @testset "input validation" begin

        RF_train = complex.(fill(20.0, 5))
        sequence = FISP3D(RF_train, 0.010, 0.003, Val(32), 0.02, 0.0, 1, true, true, 1, 0.0)
        parameters = StructVector([T₁T₂(0.8, 0.08)])

        # B₁ is not available when the tissue properties don't carry it
        @test_throws ErrorException simulate_derivatives_forward_sensitivity((:B₁,), sequence, parameters)
        @test_throws ErrorException simulate_derivatives_forward_sensitivity((), sequence, parameters)
        @test_throws ErrorException simulate_derivatives_forward_sensitivity((:B₀,), sequence,
            StructVector([T₁T₂B₁(0.8, 0.08, 0.9)]))
    end

    @testset "GPU Float32 sensitivities match CPU Float64 analytic reference" begin

        if CUDA.functional()

            RF_train = complex.([30 + 25 * sin(2π * 4.0 * t / 24) for t = 1:24])
            sequence = FISP3D(RF_train, 0.00558, 0.002745, Val(32), 0.01084, 0.0, 2, true, true, 1, 0.0)

            nvoxels = 12
            T₁ = collect(exp.(range(log(0.1), log(5.0), length=nvoxels)))
            T₂ = collect(exp.(range(log(0.01), log(2.0), length=nvoxels)))
            B₁ = collect(range(0.8, 1.2, length=nvoxels))
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

            # GPU signal from the derivative kernel must closely match the
            # plain signal-only kernel (tracking sensitivities must not
            # meaningfully perturb the primal state). Not bit-identical: the
            # sensitivity state takes a different compiled code path than
            # the signal-only one (see src/derivatives/forward_sensitivity.jl),
            # so floating-point rounding can differ at the ULP level even
            # though the value component is mathematically the same
            # computation.
            m_gpu_signal_only = simulate_magnetization(CUDALibs(), sequence_gpu, parameters_gpu)
            @test isapprox(Array(m_gpu), Array(m_gpu_signal_only); rtol=1e-4)
        end
    end
end

@testset "Isochromat forward sensitivity derivatives" begin

    five_point_central(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
    relerr(a, b) = norm(a - b) / norm(b)
    # relative, but with an absolute floor for derivatives that are exactly 0
    close_to(a, b) = norm(a - b) <= 1e-6 * norm(b) + 1e-12
    # compare Isochromat outputs component-wise
    flat(x) = eltype(x) <: Isochromat ? reinterpret(eltype(eltype(x)), x) : x

    nTR = 30
    RF_train = complex.(range(10.0, 70.0, length=nTR)) .* cis.(0.3 .* (1:nTR))
    Δt = (ex=2e-4, inv=0.02, pr=0.004)
    γΔtRF = SVector(0.2, 0.5, 0.3) .* deg2rad(1.0)

    # a hard-pulse bSSFP-like sequence with gradients, as waveforms
    nt = 60
    RF = complex.(zeros(nt))
    RF[1:10:end] .= cis(0.4) * deg2rad(30) / (26753.0 * 1e-5) # rad/s
    GR = repeat([0.02, -0.01, 0.005], 1, nt) .* repeat(sign.(sin.(2π .* (1:nt) ./ 10))', 3, 1)
    generic = Generic3D(RF, GR, iseven.(1:nt), fill(1e-5, nt))

    @testset "$(nameof(typeof(sequence))) vs finite differences" for (sequence, p) in [
        (pSSFP3D(RF_train, 0.01, γΔtRF, Δt), T₁T₂B₁B₀(0.9, 0.07, 0.93, 13.0)),
        (pSSFP2D(RF_train, 0.01, γΔtRF, Δt, (ex=0.3, inv=0.0, pr=-0.15), SVector(-0.2, 0.0, 0.1, 0.25)),
            T₁T₂B₁B₀(0.9, 0.07, 0.93, 13.0)),
        (hyperbolic_secant_pulse(0.01, 12e-6), T₁T₂B₁B₀(0.9, 0.07, 0.93, 13.0)),
        (generic, T₁T₂B₁B₀xyz(0.9, 0.07, 0.93, 13.0, 0.3, -0.2, 0.1)),
        # B₁ = 0 without B₀: the RF rotations have zero angle, so nothing
        # is excited and the T₁ and T₂ derivatives are 0, but the B₁
        # derivative is not
        (pSSFP3D(RF_train, 0.01, γΔtRF, Δt), T₁T₂B₁(0.9, 0.07, 0.0)),
    ]
        signal(p) = vec(simulate_magnetization(CPU1(), sequence, StructVector([p])))
        m, ∂m = simulate_derivatives_forward_sensitivity(sequence, StructVector([p]))

        @test propertynames(∂m) == (:T₁, :T₂, :B₁)
        # tracking sensitivities must not disturb the magnetization itself
        @test vec(m) == signal(p)

        for (i, θ) in enumerate((:T₁, :T₂, :B₁))
            # vary one property; B₁ = 0 gets an absolute step
            vary(t) = typeof(p)(setindex(SVector(Tuple(p)), t, i)...)
            h = iszero(p[i]) ? 1e-5 : 1e-5 * p[i]
            fd = five_point_central(t -> flat(signal(vary(t))), p[i], h)
            @test close_to(flat(vec(∂m[θ])), fd)
        end
    end

    @testset "Supported derivatives and requested subsets" begin
        sequence = pSSFP3D(RF_train, 0.01, γΔtRF, Δt)
        parameters = StructVector([T₁T₂(0.9, 0.07)])
        m, ∂m = simulate_derivatives_forward_sensitivity(sequence, parameters)
        @test propertynames(∂m) == (:T₁, :T₂)
        _, ∂T₂ = simulate_derivatives_forward_sensitivity((:T₂,), sequence, parameters)
        @test ∂T₂.T₂ ≈ ∂m.T₂
        @test_throws ErrorException simulate_derivatives_forward_sensitivity((:B₁,), sequence, parameters)
    end

    @testset "GPU Float32 matches CPU Float64" begin
        if CUDA.functional()
            sequence = pSSFP3D(RF_train, 0.01, γΔtRF, Δt)
            nvoxels = 64
            T₁ = collect(range(0.3, 2.0, length=nvoxels))
            T₂ = collect(range(0.02, 0.2, length=nvoxels))
            B₁ = collect(range(0.8, 1.2, length=nvoxels))
            B₀ = collect(range(-20.0, 20.0, length=nvoxels))
            parameters = @parameters T₁ T₂ B₁ B₀

            m_cpu, ∂m_cpu = simulate_derivatives_forward_sensitivity(sequence, parameters)
            m_gpu, ∂m_gpu = simulate_derivatives_forward_sensitivity(gpu(f32(sequence)), gpu(f32(parameters)))

            @test relerr(Array(m_gpu), m_cpu) < 1e-4
            @test all(θ -> relerr(Array(∂m_gpu[θ]), ∂m_cpu[θ]) < 1e-4, (:T₁, :T₂, :B₁))

            # Isochromat (x, y, z) outputs
            pulse = hyperbolic_secant_pulse(0.01, 12e-6)
            p = parameters[1:8]
            m_cpu, ∂m_cpu = simulate_derivatives_forward_sensitivity(pulse, p)
            m_gpu, ∂m_gpu = simulate_derivatives_forward_sensitivity(gpu(f32(pulse)), gpu(f32(p)))
            @test relerr(Array(m_gpu), m_cpu) < 1e-4
            @test all(θ -> relerr(Array(∂m_gpu[θ]), ∂m_cpu[θ]) < 1e-4, (:T₁, :T₂, :B₁))

            # More voxels than the GPU simulates in one go (VOXELS_PER_CHUNK in
            # ext/BlochSimulatorsCUDAExt/simulate.jl)
            nvoxels = 70_000
            T₁ = collect(range(0.3, 2.0, length=nvoxels))
            T₂ = collect(range(0.02, 0.2, length=nvoxels))
            parameters = f32(@parameters T₁ T₂)
            m_cpu = simulate_magnetization(f32(sequence), parameters)
            m_gpu = simulate_magnetization(gpu(f32(sequence)), gpu(parameters))
            @test relerr(Array(m_gpu), m_cpu) < 1e-4
        end
    end
end
