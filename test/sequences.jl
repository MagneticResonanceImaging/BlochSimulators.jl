@testset "Shortening sequences with getindex" begin

    # A shortened sequence must simulate exactly the first part of the full
    # one (FISP3D with a single repetition, since each repetition replays the
    # whole, shortened, train)
    nTR = 20
    RF_train = complex.(range(10.0, 60.0, length=nTR))
    T₁, T₂ = [0.8, 1.2], [0.08, 0.1]
    parameters = @parameters T₁ T₂
    Δt = (ex=1e-4, inv=0.02, pr=0.005)
    γΔtRF = @SVector fill(deg2rad(1.0) / 3, 3)

    sequences = [
        FISP2D(RF_train, rand(ComplexF64, nTR, 3), 0.010, 0.005, Val(32), 0.1, 2π * 1000),
        FISP3D(RF_train, 0.010, 0.005, Val(32), 0.02, 0.1, 1, true, true, 1, 0.0),
        pSSFP2D(RF_train, 0.010, γΔtRF, Δt, (ex=0.1, inv=0.0, pr=-0.05), SVector(-0.1, 0.0, 0.1)),
        pSSFP3D(RF_train, 0.010, γΔtRF, Δt),
    ]

    for sequence in sequences
        short = sequence[1:10]
        @test typeof(short) == typeof(sequence)
        @test length(short.RF_train) == 10
        @test simulate_magnetization(CPU1(), short, parameters) ≈
              simulate_magnetization(CPU1(), sequence, parameters)[1:10, :]
    end
end
