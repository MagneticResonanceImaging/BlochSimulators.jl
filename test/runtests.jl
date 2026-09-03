using Test
using BlochSimulators

# load general packages
using ComputationalResources
using CUDA
using Distributed
using DistributedArrays
using LinearAlgebra
using StaticArrays
using StructArrays


function make_T₁T₂_structarray(nvoxels)
    T₁ = rand(nvoxels)
    T₂ = 0.1 * T₁
    return @parameters T₁ T₂
end

@testset "Test operator functions for isochromat model" begin

    # create single spin isochromat
    m = BlochSimulators.Isochromat(rand(3)...)

    # check initial conditions
    m = BlochSimulators.initial_conditions(m)
    @test m == BlochSimulators.Isochromat(0.0, 0.0, 1.0)

    # test inversion
    # "adiabatic inversion"
    @test BlochSimulators.invert(m) == BlochSimulators.Isochromat(0.0, 0.0, -1.0)

    # "
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 1.0)
    @test BlochSimulators.invert(m, p) == BlochSimulators.Isochromat(0.0, 0.0, -1.0)

    # failed 90 degree inversion
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.5)
    @test abs(BlochSimulators.invert(m, p).z) < eps()

    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.0)
    @test BlochSimulators.invert(m, p).z == 1.0

    # test decay
    m = BlochSimulators.Isochromat(1.0, 1.0, -1.0)
    Δt = 1e7
    E₁ = exp(-Δt / 0.8)
    E₂ = exp(-Δt / 0.05)
    # really long Δt so everything should be zero after decay
    m = BlochSimulators.decay(m, E₁, E₂)
    @test all(iszero.(m))

    # test regrowth

    m = BlochSimulators.regrowth(m, E₁)
    # really long Δt so z component should be 1
    @test m == BlochSimulators.initial_conditions(m)

    # rotate

    z = 0.0
    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 0.0)
    γΔtGRz = 0.0
    Δt = 0.0

    m = BlochSimulators.Isochromat(0.0, 0.0, 1.0)

    # rotate over x-axis
    γΔtRF = π / 2 + 0.0im
    mx = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test mx.y == -1.0 && abs(mx.z) < eps()

    # rotate over y-axis
    γΔtRF = 0.0 + im * π / 2
    my = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test my.x == -1.0 && abs(my.z) < eps()

    # rotate over z-axis
    γΔtRF = 0.0 + 0.0 * im
    γΔtGRz = π / 2
    z = 1.0
    m = BlochSimulators.Isochromat(1.0, 0.0, 0.0)
    mz = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test mz.y == -1.0 && abs(mz.x) < eps()

    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 1.0)
    γΔtGRz = 0.0
    Δt = 0.5
    mz = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test mz.x == -1.0 && abs(mz.y) < eps()

    # also test the rotate function without RF argument

    γΔtGRz = π / 2
    z = 1.0
    m = BlochSimulators.Isochromat(1.0, 0.0, 0.0)
    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 0.0)
    mz = BlochSimulators.rotate(m, γΔtGRz, z, Δt, p)
    @test mz.y == -1.0 && abs(mz.x) < eps()

    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 1.0)
    γΔtGRz = 0.0
    Δt = 0.5
    mz = BlochSimulators.rotate(m, γΔtGRz, z, Δt, p)
    @test mz.x == -1.0 && abs(mz.y) < eps()

end

# test some individual functions in BlochSimulators
@testset "Test operator functions for EPG model" begin

    @testset "state matrix mutation interface" begin
        R = @SMatrix [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]
        mutable_states = ConfigurationStates(ones(ComplexF64, 3, 2))
        static_states = BlochSimulators.ConfigurationStatesSubset(
            @SMatrix ones(ComplexF64, 3, 2)
        )

        for Ω in (mutable_states, static_states)
            @test mul!(Ω, R, Ω) === Ω
            @test Ω == [2 2; 3 3; 4 4]
            @test fill!(Ω, 0) === Ω
            @test all(iszero, Ω)
            Ω .= 1
            Ω .*= (2, 3, 4)
            @test Ω == [2 2; 3 3; 4 4]
        end
    end

    # create single spin isochromat
    Ω = zeros(ComplexF64, 3, 20) |> ConfigurationStates

    # check initial conditions
    BlochSimulators.initial_conditions!(Ω)
    @test Ω[3, 1] == 1.0 + 0.0im
    @test sum(Ω) == 1.0 + 0.0im

    # test inversion

    # "adiabatic inversion"
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω)
    @test Ω[3, 1] == -1

    # "non-adiabatic inversion"
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 1.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω, p)
    @test Ω[3, 1] == -1

    # failed 90 degree inversion
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.5)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω, p)
    @test abs(Ω[3, 1]) < eps()

    # 0 B1 so 0 effect
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω, p)
    @test Ω[3, 1] == 1

    # check "adiabatic inversion" for higher order states as well
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.Z(Ω) .= 0.1
    BlochSimulators.invert!(Ω)
    @test all(Ω[3, :] .== complex(-0.1))

    # test decay
    Ω = rand(ComplexF64, 3, 20) |> ConfigurationStates
    Δt = 1e7
    E₁ = exp(-Δt / 0.8)
    E₂ = exp(-Δt / 0.05)
    # really long Δt so everything should be zero after decay
    BlochSimulators.decay!(Ω, E₁, E₂)
    @test all(iszero.(Ω.matrix))

    # test regrowth

    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.decay!(Ω, E₁, E₂)
    BlochSimulators.regrowth!(Ω, E₁)
    # really long Δt so z component should be 1
    @test Ω[3, 1] == 1.0

    # rotate

    RF = complex(90.0)
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 1.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.excite!(Ω, RF, p)
    @test Ω[1, 1] == -im && Ω[2, 1] == conj(Ω[1, 1])

    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.excite!(Ω, complex(45.0), p)
    @test Ω[1, 1] ≈ -(√2 / 2) * im && Ω[2, 1] == conj(Ω[1, 1])

    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.excite!(Ω, complex(90.0), p)
    @test Ω[1, 1] ≈ 0 && Ω[2, 1] == conj(Ω[1, 1])

    # dephasing

    # dont act on Z states
    Ω = rand(ComplexF64, 3, 20) |> ConfigurationStates
    Ω2 = copy(Ω)
    BlochSimulators.dephasing!(Ω)
    @test Ω[3, :] == Ω2[3, :]

    # check "left boundary"
    BlochSimulators.initial_conditions!(Ω)
    Ω[1, 1] = 1
    Ω[2, 1] = 2
    Ω[2, 2] = 3im
    BlochSimulators.dephasing!(Ω)
    @test Ω[1, 2] == 1 && Ω[2, 1] == 3im && Ω[1, 1] == conj(Ω[2, 1])

    # check "right boundary"
    Ω .= reshape(1:60, 3, 20)
    Ω[2, 1] = 1
    Ω[2, 2] = 5im
    Ωpre = copy(Ω)
    BlochSimulators.dephasing!(Ω)
    @test Ω[1, end] == Ωpre[1, end-1]
    @test Ω[2, end-1] == Ωpre[2, end]
    @test Ω[3, end] == Ωpre[3, end]

    # spoil

    # check that transverse states are nulled
    Ω = rand(ComplexF64, 3, 20) |> ConfigurationStates
    BlochSimulators.spoil!(Ω)
    @test all(Ω[1, :] .== complex(0.0))
    @test all(Ω[2, :] .== complex(0.0))

    # diffusion
    Ω_in = ones(ComplexF64, 3, 20) |> ConfigurationStates
    Ω = copy(Ω_in) |> ConfigurationStates

    D_dispersion = 0.0
    no_diffusion_matrix = BlochSimulators.diffusion_decay_matrix(Ω, D_dispersion)

    # with no diffusion, diffuse!() has no effect
    BlochSimulators.diffuse!(Ω, no_diffusion_matrix)
    rms_diff = sqrt(sum(abs2.(Ω - Ω_in)) / length(Ω))
    @test rms_diff < 1e-8

    # with diffusion, diffuse!() reduces the states
    Ω = copy(Ω_in) |> ConfigurationStates
    some_diffusion_matrix = BlochSimulators.diffusion_decay_matrix(Ω, 0.1)

    @test all(some_diffusion_matrix .<= 1.0)

    BlochSimulators.diffuse!(Ω, some_diffusion_matrix)
    nonzero_in = abs.(Ω_in[:, 2:end])
    nonzero_out = abs.(Ω[:, 2:end])
    @test all(nonzero_out .< nonzero_in)

    # The effect on high states is expected to be larger than the effect on low states
    prev_col = some_diffusion_matrix[:,1]
    diffusion_decay_larger_for_higher_order = true
    for i in 2:20
        if any(some_diffusion_matrix[:,i] .> prev_col)
            diffusion_decay_larger_for_higher_order = false
        end
        prev_col = some_diffusion_matrix[:,i]
    end
    @test diffusion_decay_larger_for_higher_order


end

@testset "Test functions to change precision" begin

    # f32 and f64 should recursively go through nested structures and convert floating point numbers only

    # first, test some complicated but random nested structure
    x = [1, 2.0, 3.0f0, [4.0, 5.0], 6.0im, (7.0, 8.0im), (a=9.0, b=10.0im)]

    @test f64(x) == x
    @test f32(x) == [1, 2.0f0, 3.0f0, [4.0f0, 5.0f0], 6.0f0im, (7.0f0, 8.0f0im), (a=9.0f0, b=10.0f0im)]

    # test FISP sequence struct
    nTR = 10
    s = FISP2D(nTR)

    @test f64(s) == s

    @test f32(s).RF_train == ComplexF32.(s.RF_train)
    @test f32(s).sliceprofiles == ComplexF32.(s.sliceprofiles)
    @test f32(s).TR == Float32(s.TR)
    @test f32(s).TE == Float32(s.TE)
    @test f32(s).TI == Float32(s.TI)
    @test f32(s).max_state == s.max_state

    # test Cartesian trajectory struct
    t = CartesianTrajectory2D(nTR, 100)

    @test f64(t) == t

    @test f32(t).nreadouts == t.nreadouts
    @test f32(t).nsamplesperreadout == t.nsamplesperreadout
    @test f32(t).Δt == Float32(t.Δt)
    @test f32(t).k_start_readout == ComplexF32.(t.k_start_readout)
    @test f32(t).Δk_adc == ComplexF32.(t.Δk_adc)
    @test f32(t).py == t.py

    # test AbstractTissueProperties
    p = T₁T₂B₁B₀(1.0, 2.0, 3.0, 4.0)
    f32(p) == T₁T₂B₁B₀(1.0f0, 2.0f0, 3.0f0, 4.0f0)
    f64(f32(p)) == p

    # test SimulationParameters
    nvoxels = 100
    T₁ = rand(nvoxels)
    T₂ = 0.1 * T₁
    parameters = @parameters T₁ T₂

    T₁ = f32(T₁)
    T₂ = f32(T₂)
    parameters_f32 = @parameters T₁ T₂
    @test parameters_f32 == f32(parameters)

    # test StructArray{<:Coordinates}
    x, y, z = rand(nvoxels), rand(nvoxels), rand(nvoxels)
    coordinates = @coordinates x y z

    x = f32(x)
    y = f32(y)
    z = f32(z)
    coordinates_f32 = @coordinates x y z
    @test coordinates_f32 == f32(coordinates)
end

@testset "Test functions to move to gpu" begin

    if CUDA.functional()
        # first, test some complicated but random nested structure
        x = [1, 2.0, 3.0f0, [4.0, 5.0], 6.0im, (7.0, 8.0im), (a=9.0, b=10.0im)]

        @test gpu(x) == [1, 2.0, 3.0f0, CuArray([4.0, 5.0]), 6.0im, (7.0, 8.0im), (a=9.0, b=10.0im)]

        # test FISP sequence struct
        nTR = 10
        s = FISP2D(nTR)

        @test gpu(s).RF_train == CuArray(s.RF_train)
        @test gpu(s).sliceprofiles == CuArray(s.sliceprofiles)
        @test gpu(s).TR == s.TR
        @test gpu(s).TE == s.TE
        @test gpu(s).TI == s.TI
        @test gpu(s).max_state == s.max_state

        # test Cartesian trajectory struct
        t = CartesianTrajectory2D(nTR, 100)

        @test gpu(t).nreadouts == t.nreadouts
        @test gpu(t).nsamplesperreadout == t.nsamplesperreadout
        @test gpu(t).Δt == t.Δt
        @test gpu(t).k_start_readout == CuArray(t.k_start_readout)
        @test gpu(t).Δk_adc == t.Δk_adc
        @test gpu(t).py == CuArray(t.py)
        @test gpu(t).readout_oversampling == t.readout_oversampling

        # test AbstractTissueProperties
        p = T₁T₂B₁B₀(1.0, 2.0, 3.0, 4.0)
        @test gpu(p) == p
        @test gpu([p]) == CuArray([p])

        # test SimulationParameters
        nvoxels = 100
        T₁ = rand(nvoxels)
        T₂ = 0.1 * T₁
        parameters = @parameters T₁ T₂

        T₁ = gpu(T₁)
        T₂ = gpu(T₂)
        parameters_gpu = @parameters T₁ T₂
        @test typeof(parameters_gpu) == typeof(gpu(parameters))
        @test CUDA.@allowscalar parameters_gpu == gpu(parameters)

        # test StructArray{<:Coordinates}
        x, y, z = rand(nvoxels), rand(nvoxels), rand(nvoxels)
        coordinates = @coordinates x y z
        coordinates_gpu = gpu(coordinates)

        xyz = Iterators.product(x, y, z) |> collect |> vec
        @test coordinates_gpu.x == gpu([r[1] for r in xyz])
        @test coordinates_gpu.y == gpu([r[2] for r in xyz])
        @test coordinates_gpu.z == gpu([r[3] for r in xyz])

        @test CUDA.@allowscalar coordinates_gpu == gpu(coordinates)

        @test_throws ArgumentError make_coordinates(gpu(x), gpu(y), gpu(z))

    end

end

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

@testset "Test for SpokesTrajectory" begin

    # Simulate magnetization at echo times for a single voxel with coordinates (0,0,0)
    nTR = 100
    sequence = FISP2D(nTR)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, 1.0, 0.0)] |> StructArray
    coordinates = [Coordinates(0.0, 0.0, 0.0)] |> StructArray

    d = simulate_magnetization(CPU1(), sequence, parameters)

    # Use some trajectory to simulate signal
    nr, ns = 100, 40
    trajectory = CartesianTrajectory2D(nr, ns)

    s = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates)

    # Because this voxel has x = y = 0, a gradient trajectory
    # should not influence it and for one voxel there's no
    # volume integral either
    @test d ≈ vec(s[(ns÷2)+1, :])

    # If we now simulate with a voxel with x and y non-zero, then
    # at echo time the magnetization should be the same in abs
    s2 = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates)

    @test abs.(d) ≈ abs.(vec(s2[(ns÷2)+1, :]))

end

@testset "Tests for CartesianTrajectory2D" begin

    # constants
    nr = 100
    ns = 128
    Δt = 4e-6 # s
    fovx = 10.0 # cm
    fovy = 10.0 # cm
    Δkˣ = 2π / fovx
    Δkʸ = 2π / fovy
    py = collect(-50:49)
    k0 = [(-ns / 2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nr]
    os = 1

    # assemble Cartesian trajectory
    cartesian = CartesianTrajectory2D(nr, ns, Δt, k0, Δkˣ, py, os)

    # test whether getindex method to reduce sequence length works
    @test cartesian[1:50].k_start_readout == CartesianTrajectory2D(50, ns, Δt, k0[1:50], Δkˣ, py[1:50], os).k_start_readout
    @test cartesian[1:50].Δk_adc == CartesianTrajectory2D(50, ns, Δt, k0[1:50], Δkˣ, py[1:50], os).Δk_adc
    @test cartesian[1:50].py == CartesianTrajectory2D(50, ns, Δt, k0[1:50], Δkˣ, py[1:50], os).py

    @test BlochSimulators.sampling_mask(cartesian)[10] == CartesianIndices((1:ns, 10:10))

    @test BlochSimulators.kspace_coordinates(cartesian)[:, 1] == k0[1] .+ (collect(0:ns-1) .* Δkˣ)
end

@testset "Tests for RadialTrajectory2D" begin

    # assemble radial trajectory
    nr = 100
    ns = 128
    Δt = 4e-6 # s
    os = 2  # factor two oversampling
    fovx = 10.0 # cm
    fovy = 10.0 # cm
    φ = π / ((√5 + 1) / 2) # golden angle of ~111 degrees
    Δkˣ = 2π / (os * fovx)
    φ = collect(φ .* (0:nr-1))
    k0 = -(ns / 2) * Δkˣ + 0.0im
    k0 = collect(@. exp(im * φ) * k0)
    Δk = Δkˣ + 0.0im
    Δk = collect(@. exp(im * φ) * Δk)
    os = 1

    radial = RadialTrajectory2D(nr, ns, Δt, k0, Δk, φ, 1)

    # test whether getindex method to reduce sequence length works
    @test radial[1:50].k_start_readout == RadialTrajectory2D(50, ns, Δt, k0[1:50], Δk[1:50], φ[1:50], os).k_start_readout
    @test radial[1:50].Δk_adc == RadialTrajectory2D(50, ns, Δt, k0[1:50], Δk[1:50], φ[1:50], os).Δk_adc
    @test radial[1:50].φ == RadialTrajectory2D(50, ns, Δt, k0[1:50], Δk[1:50], φ[1:50], os).φ

    # test gradient delay for radial
    S = rand(2, 2)
    radial_delay = deepcopy(radial)
    BlochSimulators.add_gradient_delay!(radial_delay, S)
    @test all(radial_delay.k_start_readout .== BlochSimulators.add_gradient_delay(radial, S).k_start_readout)

end

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
