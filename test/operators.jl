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
