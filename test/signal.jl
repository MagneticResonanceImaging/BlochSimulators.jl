@testset "Signal simulation sanity checks" begin

    # if magnetization .= 1, x and y .= 0, coil_sensitivities .= 1, T₂ .= Inf,
    # then signal should simply be the nr of voxels at all time points
    nv = 100 # voxels
    nr = 100 # readouts
    ns = 10  # samples per readout
    nc = 1 # number of coils

    magnetization = complex.(ones(nr, nv))
    parameters = fill(T₁T₂ρˣρʸ(Inf, Inf, 1.0, 0.0), nv) |> StructArray
    coordinates = fill(Coordinates(0.0, 0.0, 0.0), nv) |> StructArray
    trajectory = RadialTrajectory2D(nr, ns)
    coil_sensitivities = complex(ones(nv, nc))
    resource = CPU1()

    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    @test signal == fill(nv, ns * nr, nc)

    # if proton density is 0, then signal should be 0

    parameters = fill(T₁T₂ρˣρʸ(rand(), rand(), 0.0, 0.0), nv) |> StructArray

    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    @test signal == zeros(ns * nr, nc)

    # if coil sensitivities are 0 everywhere, then signal should be 0

    parameters = fill(T₁T₂ρˣρʸ(rand(4)...), nv) |> StructArray
    nc = 4
    coil_sensitivities = complex(zeros(nv, nc))
    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    @test signal == zeros(ns * nr, nc)

end

@testset "Test Cartesian signal simulations on different computational resources" begin

    # Simulate signal with CPU1() as reference
    nTR = 1000
    nvoxels = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, rand(2)...) for _ = 1:nvoxels] |> StructArray

    trajectory = CartesianTrajectory2D(nTR, 100)
    nc = 2
    coil_sensitivities = rand(ComplexF64, nvoxels, nc)
    coordinates = [Coordinates(rand(3)...) for _ = 1:nvoxels] |> StructArray
    signal_cpu1 = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates, coil_sensitivities)

    # Now simulate with CPUThreads() (multi-threaded CPU) and check if outcome is the same
    signal_cputhreads = simulate_signal(CPUThreads(), sequence, parameters, trajectory, coordinates, coil_sensitivities)
    @test signal_cpu1 ≈ signal_cputhreads

    # # Now add workers and simulate with CPUProcesses() (distributed CPU)
    # # and check if outcome is the same
    # if workers() == [1]
    #     addprocs(2, exeflags="--project=.")
    #     @everywhere using BlochSimulators, ComputationalResources, DistributedArrays
    # end

    # signal_cpuprocesses = simulate_signal(CPUProcesses(), sequence, distribute(parameters), trajectory, distribute(coordinates), distribute(coil_sensitivities))
    # @test signal_cpu1 ≈ signal_cpuprocesses

    if CUDA.functional()
        # Simulate with CUDALibs() (GPU) and check if outcome is the same
        signal_cudalibs = simulate_signal(CUDALibs(), gpu(sequence), gpu(parameters), gpu(trajectory), gpu(coordinates), gpu(coil_sensitivities))
        @test signal_cpu1 ≈ convert(Array, signal_cudalibs)
    end

end

@testset "Test radial signal simulations on different computational resources" begin

    # Simulate signal with CPU1() as reference
    nTR = 1000
    nvoxels = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, rand(2)...) for _ = 1:nvoxels] |> StructArray

    trajectory = RadialTrajectory2D(nTR, 100)
    nc = 2
    coil_sensitivities = rand(ComplexF64, nvoxels, nc)
    coordinates = [Coordinates(rand(3)...) for _ = 1:nvoxels] |> StructArray
    signal_cpu1 = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates, coil_sensitivities)

    # Now simulate with CPUThreads() (multi-threaded CPU) and check if outcome is the same
    signal_cputhreads = simulate_signal(CPUThreads(), sequence, parameters, trajectory, coordinates, coil_sensitivities)
    @test signal_cpu1 ≈ signal_cputhreads

    # # Now add workers and simulate with CPUProcesses() (distributed CPU)
    # # and check if outcome is the same
    # if workers() == [1]
    #     addprocs(2, exeflags="--project=.")
    #     @everywhere using BlochSimulators, ComputationalResources, DistributedArrays
    # end

    # signal_cpuprocesses = simulate_signal(CPUProcesses(), sequence, distribute(parameters), trajectory, distribute(coordinates), distribute(coil_sensitivities))
    # @test signal_cpu1 ≈ signal_cpuprocesses

    if CUDA.functional()
        # Simulate with CUDALibs() (GPU) and check if outcome is the same
        signal_cudalibs = simulate_signal(CUDALibs(), gpu(sequence), gpu(parameters), gpu(trajectory), gpu(coordinates), gpu(coil_sensitivities))
        @test signal_cpu1 ≈ convert(Array, signal_cudalibs)
    end

end

@testset "Test simulation of signal in batches" begin

    # Simulate signal with CPU1() as reference
    nTR = 1000
    nvoxels = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, rand(2)...) for _ = 1:nvoxels] |> StructArray

    trajectory = CartesianTrajectory2D(nTR, 100)
    nc = 2
    coil_sensitivities = rand(ComplexF64, nvoxels, nc)
    coordinates = [Coordinates(rand(3)...) for _ = 1:nvoxels] |> StructArray
    signal_reference = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates, coil_sensitivities)

    # Partition the voxels in batches
    num_voxels_per_partition = 120
    partition_idx = Iterators.partition(1:nvoxels, num_voxels_per_partition)

    partitioned_parameters = [parameters[idx] for idx in partition_idx]
    partitioned_coordinates = [coordinates[idx] for idx in partition_idx]
    partitioned_coil_sensitivities = [coil_sensitivities[idx, :] for idx in partition_idx]

    # Simulate signal in batches
    signal_batched = simulate_signal(CPU1(), sequence, partitioned_parameters, trajectory, partitioned_coordinates, partitioned_coil_sensitivities)

    # Test if the signal is the same as the reference signal
    @test signal_reference ≈ signal_batched

    if CUDA.functional()
        # Simulate with CUDALibs()
        signal_reference = simulate_signal(CUDALibs(), gpu(sequence), gpu(parameters), gpu(trajectory), gpu(coordinates), gpu(coil_sensitivities))

        # Simulate signal in batches
        signal_batched = simulate_signal(CUDALibs(), gpu(sequence), gpu(partitioned_parameters), gpu(trajectory), gpu(partitioned_coordinates), gpu(partitioned_coil_sensitivities))

        # Test if the signal is the same as the reference signal
        @test signal_reference ≈ signal_batched
    end

end
