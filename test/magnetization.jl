@testset "Test dictionary generation on different computational resources" begin

    # Simulate dictionary with CPU1()
    nTR = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    nvoxels = 100
    parameters = make_T₁T₂_structarray(nvoxels)

    magnetization_cpu1 = simulate_magnetization(CPU1(), sequence, parameters)

    # Now simulate with CPUThreads() (multi-threaded CPU) and check if outcome is the same
    magnetization_cputhreads = simulate_magnetization(CPUThreads(), sequence, parameters)
    @test magnetization_cpu1 ≈ magnetization_cputhreads

    # # Now add workers and simulate with CPUProcesses() (distributed CPU)
    # # and check if outcome is the same
    # if workers() == [1]
    #     addprocs(2, exeflags="--project=.")
    #     @everywhere using BlochSimulators, ComputationalResources
    # end

    # magnetization_cpuprocesses = simulate_magnetization(CPUProcesses(), sequence, distribute(parameters))

    # @test magnetization_cpu1 ≈ convert(Array, magnetization_cpuprocesses)

    if CUDA.functional()
        # Simulate with CUDALibs() (GPU) and check if outcome is the same
        magnetization_cudalibs = simulate_magnetization(CUDALibs(), gpu(sequence), gpu(parameters)) |> collect
        @test magnetization_cpu1 ≈ magnetization_cudalibs
    end

end

@testset "Test single voxel simulations" begin

    # Methods have been added that accept a single `<:AbstractTissueParameters` to perform simulations in a single voxel. The results should be the same as if the tissue properties are wrapped in a `StructVector` and the methods for batch simulations are used.
    sequence = FISP2D(10)
    tissue_properties = T₁T₂B₁B₀(1.0, 0.1, 0.9, 10.0)
    parameters = StructVector([tissue_properties])

    m₁ = simulate_magnetization(sequence, tissue_properties)
    m₂ = simulate_magnetization(sequence, parameters)

    @test m₁ == m₂

    m₁,∂m₁ = simulate_derivatives_finite_difference(sequence, tissue_properties)
    m₂,∂m₂ = simulate_derivatives_finite_difference(sequence, parameters)

    @test m₁ == m₂
    @test ∂m₁ == ∂m₂
end

@testset "Test diffusion code" begin

    nTR = 1000;
    nvoxels = 5;
    sequence = FISP2D(nTR);
    RF_train = complex.([30+25*sin(2π*4.0*t/nTR) for t = 1:nTR]);
    slice_corrections = rand(ComplexF64, nTR, 3);
    Δk_spoil = 2π*1000

    # FISP sequence with spoiling gradients that can introduce diffusion effects
    sequence = FISP2D(RF_train, slice_corrections, 0.010, 0.005, Val(32), 0.1, Δk_spoil)

    # simulate magnetization without diffusion
    parameters = [T₁T₂ρˣρʸ(v*1.0, v*0.1, 1.0, 0.0) for v = 1:nvoxels] |> StructArray;
    m_no_diffusion   = simulate_magnetization(CPU1(), sequence, parameters);

    # simulate magnetization with D=0
    parameters = [T₁T₂Dρˣρʸ(v*1.0, v*0.1, 0.0, 1.0, 0.0) for v = 1:nvoxels] |> StructArray;
    m_zero_diffusion = simulate_magnetization(CPU1(), sequence, parameters);

    parameters = [T₁T₂Dρˣρʸ(v*1.0, v*0.1, v*0.001, 1.0, 0.0) for v = 1:nvoxels] |> StructArray;
    m_some_diffusion = simulate_magnetization(CPU1(), sequence, parameters);

    # test diffusion simulation to not affect result when D=0
    @test m_no_diffusion == m_zero_diffusion
    # test that diffusion with non-zero D affects the result
    @test m_some_diffusion !== m_zero_diffusion

    if CUDA.has_cuda_gpu()
        # Check that CPU and GPU implementations give the same results
        parameters = [T₁T₂Dρˣρʸ(v*1.0, v*0.1, v * 0.01, 1.0, 0.0) for v = 1:nvoxels] |> StructArray;

        d_nonzero_cpu = simulate_magnetization(CPU1(), f32(sequence), f32(parameters));
        d_nonzero_gpu = simulate_magnetization(CUDALibs(), gpu(f32(sequence)), gpu(f32(parameters))) |> collect;
        @test d_nonzero_cpu ≈ d_nonzero_gpu
    end
end
