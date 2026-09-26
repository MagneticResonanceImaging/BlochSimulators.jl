@testset "Test finite differences: derivative tests" begin


    stepsizes = T₁T₂B₁B₀(1e-4, 1e-4, 1e-4, 1e-4)

    sequence = FISP2D(10)
    parameters = StructVector([T₁T₂B₁B₀(1.0, 0.1, 0.9, 10.0)])

    m = simulate_magnetization(CPU1(), sequence, parameters)

    # Single derivative tests
    ∂m∂T₁ = BlochSimulators.finite_difference_single_tissue_property(:T₁, m, sequence, parameters, stepsizes)
    ∂m∂T₂ = BlochSimulators.finite_difference_single_tissue_property(:T₂, m, sequence, parameters, stepsizes)
    ∂m∂B₁ = BlochSimulators.finite_difference_single_tissue_property(:B₁, m, sequence, parameters, stepsizes)
    ∂m∂B₀ = BlochSimulators.finite_difference_single_tissue_property(:B₀, m, sequence, parameters, stepsizes)

    @test size(∂m∂T₁) == size(m)
    @test size(∂m∂T₂) == size(m)
    @test size(∂m∂B₁) == size(m)
    @test size(∂m∂B₀) == size(m)

    parameters_with_ΔT₁ = StructVector([T₁T₂B₁B₀(1.0 + 1e-4, 0.1, 0.9, 10.0)])
    parameters_with_ΔT₂ = StructVector([T₁T₂B₁B₀(1.0, 0.1 + 1e-4, 0.9, 10.0)])
    parameters_with_ΔB₁ = StructVector([T₁T₂B₁B₀(1.0, 0.1, 0.9 + 1e-4, 10.0)])
    parameters_with_ΔB₀ = StructVector([T₁T₂B₁B₀(1.0, 0.1, 0.9, 10.0 + 1e-4)])

    m_with_ΔT₁ = simulate_magnetization(CPU1(), sequence, parameters_with_ΔT₁)
    m_with_ΔT₂ = simulate_magnetization(CPU1(), sequence, parameters_with_ΔT₂)
    m_with_ΔB₁ = simulate_magnetization(CPU1(), sequence, parameters_with_ΔB₁)
    m_with_ΔB₀ = simulate_magnetization(CPU1(), sequence, parameters_with_ΔB₀)

    @test ∂m∂T₁ ≈ (m_with_ΔT₁ - m) / 1e-4
    @test ∂m∂T₂ ≈ (m_with_ΔT₂ - m) / 1e-4
    @test ∂m∂B₁ ≈ (m_with_ΔB₁ - m) / 1e-4
    @test ∂m∂B₀ ≈ (m_with_ΔB₀ - m) / 1e-4

    # All derivatives tests
    ∂m = BlochSimulators.simulate_derivatives_finite_difference((:T₁, :T₂, :B₁, :B₀), m, sequence, parameters, stepsizes)

    @test propertynames(∂m) == (:T₁, :T₂, :B₁, :B₀)
    @test ∂m.T₁ == ∂m∂T₁
    @test ∂m.T₂ == ∂m∂T₂
    @test ∂m.B₁ == ∂m∂B₁
    @test ∂m.B₀ == ∂m∂B₀

end

@testset "Test finite differences: step size tests" begin

    DEFAULT_STEPSIZES = BlochSimulators.DEFAULT_STEPSIZES_FINITE_DIFFERENCE

    # Test default step sizes
    Δ = BlochSimulators._calculate_stepsize(:T₁, Float64, DEFAULT_STEPSIZES)
    @test Δ == DEFAULT_STEPSIZES.T₁
    Δ = BlochSimulators._calculate_stepsize(:T₂, Float64, DEFAULT_STEPSIZES)
    @test Δ == DEFAULT_STEPSIZES.T₂
    Δ = BlochSimulators._calculate_stepsize(:B₁, Float64, DEFAULT_STEPSIZES)
    @test Δ == DEFAULT_STEPSIZES.B₁
    Δ = BlochSimulators._calculate_stepsize(:B₀, Float64, DEFAULT_STEPSIZES)
    @test Δ == DEFAULT_STEPSIZES.B₀

    # Test default step sizes with Float32
    Δ = BlochSimulators._calculate_stepsize(:T₁, Float32, DEFAULT_STEPSIZES)
    @test Δ == Float32(DEFAULT_STEPSIZES.T₁)
    Δ = BlochSimulators._calculate_stepsize(:T₂, Float32, DEFAULT_STEPSIZES)
    @test Δ == Float32(DEFAULT_STEPSIZES.T₂)
    Δ = BlochSimulators._calculate_stepsize(:B₁, Float32, DEFAULT_STEPSIZES)
    @test Δ == Float32(DEFAULT_STEPSIZES.B₁)
    Δ = BlochSimulators._calculate_stepsize(:B₀, Float32, DEFAULT_STEPSIZES)
    @test Δ == Float32(DEFAULT_STEPSIZES.B₀)

    # Test custom step sizes
    Δ = BlochSimulators._calculate_stepsize(:T₁, Float64, T₁T₂B₁B₀(1e-1, 1e-2, 1e-3, 1e-4))
    @test Δ == 1e-1
    Δ = BlochSimulators._calculate_stepsize(:T₂, Float64, T₁T₂B₁B₀(1e-1, 1e-2, 1e-3, 1e-4))
    @test Δ == 1e-2
    Δ = BlochSimulators._calculate_stepsize(:B₁, Float64, T₁T₂B₁B₀(1e-1, 1e-2, 1e-3, 1e-4))
    @test Δ == 1e-3
    Δ = BlochSimulators._calculate_stepsize(:B₀, Float64, T₁T₂B₁B₀(1e-1, 1e-2, 1e-3, 1e-4))
    @test Δ == 1e-4

    # Test error for unknown derivative type
    @test_throws ErrorException BlochSimulators._calculate_stepsize(:unknown, Float64, DEFAULT_STEPSIZES)

end

@testset "Test finite differences: out-of-place difference quotient tests" begin

    # Test case 1: Δm and m have the same size
    Δm = [1, 2, 3]
    m = [4, 5, 6]
    Δ = 2
    expected_result = [-1.5, -1.5, -1.5]
    @test BlochSimulators._finite_difference_quotient(Δm, m, Δ) ≈ expected_result

    # Test case 2: Δm and m have different sizes
    Δm = [1, 2, 3]
    m = [4, 5]
    Δ = 2
    @test_throws ErrorException BlochSimulators._finite_difference_quotient(Δm, m, Δ)

    # Test case 3: Δ is zero
    Δm = [1, 2, 3]
    m = [4, 5, 6]
    Δ = 0
    @test_throws ErrorException BlochSimulators._finite_difference_quotient(Δm, m, Δ)
end

@testset "Test finite differences: in-place difference quotient tests" begin

    # Test case 1: Δm and m have the same size
    Δm = [1.0, 2.0, 3.0]
    m = [4.0, 5.0, 6.0]
    Δ = 2
    expected_result = [-1.5, -1.5, -1.5]
    BlochSimulators._finite_difference_quotient!(Δm, m, Δ)
    @test Δm ≈ expected_result

    # Test case 2: Δm and m have different sizes
    Δm = [1, 2, 3]
    m = [4, 5]
    Δ = 2
    @test_throws ErrorException BlochSimulators._finite_difference_quotient!(Δm, m, Δ)

    # Test case 3: Δ is zero
    Δm = [1, 2, 3]
    m = [4, 5, 6]
    Δ = 0
    @test_throws ErrorException BlochSimulators._finite_difference_quotient!(Δm, m, Δ)
end
