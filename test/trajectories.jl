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
