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
