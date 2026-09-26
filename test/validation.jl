# Validation against closed-form solutions
# ========================================
# Each test runs a sequence long enough to reach its steady state (or uses a
# single excitation) and compares with the textbook expression for it. The
# forward-sensitivity derivatives are compared with (finite differences of)
# the same expressions, which checks them independently of the simulator.
# See docs/src/validation.md for the formulas and references.

five_point(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)

# Derivative of |m| from that of m
∂abs(m, ∂m) = real(conj(m) * ∂m) / abs(m)

"""
Closed-form derivatives of `f(T₁, T₂, B₁)` w.r.t. each argument, by
finite differences of the formula (which is smooth and cheap, so these are
accurate to ~1e-10).
"""
closed_form_derivatives(f, T₁, T₂, B₁) = (
    T₁=five_point(t -> f(t, T₂, B₁), T₁, 1e-5 * T₁),
    T₂=five_point(t -> f(T₁, t, B₁), T₂, 1e-5 * T₂),
    B₁=five_point(b -> f(T₁, T₂, b), B₁, 1e-5 * B₁))

# A spoiled gradient echo sequence with ideal spoiling of the transverse
# magnetization at the end of each TR.
struct IdealSpoiledGRE{T,Ns} <: EPGSimulator{T,Ns}
    α::T # degrees
    nTR::Int
    TR::T
    TE::T
    max_state::Val{Ns}
end
BlochSimulators.output_size(s::IdealSpoiledGRE) = s.nTR
BlochSimulators.output_eltype(::IdealSpoiledGRE{T}) where {T} = Complex{T}
function BlochSimulators.simulate_magnetization!(magnetization, s::IdealSpoiledGRE, Ω, p::AbstractTissueProperties)
    E₁ᵀᴱ, E₂ᵀᴱ = BlochSimulators.E₁(Ω, s.TE, p.T₁), BlochSimulators.E₂(Ω, s.TE, p.T₂)
    E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ = BlochSimulators.E₁(Ω, s.TR - s.TE, p.T₁), BlochSimulators.E₂(Ω, s.TR - s.TE, p.T₂)
    BlochSimulators.initial_conditions!(Ω)
    for TR in 1:s.nTR
        BlochSimulators.excite!(Ω, s.α, p)
        BlochSimulators.decay!(Ω, E₁ᵀᴱ, E₂ᵀᴱ)
        BlochSimulators.regrowth!(Ω, E₁ᵀᴱ)
        BlochSimulators.sample_transverse!(magnetization, TR, Ω)
        BlochSimulators.decay!(Ω, E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ)
        BlochSimulators.regrowth!(Ω, E₁ᵀᴿ⁻ᵀᴱ)
        BlochSimulators.spoil!(Ω)
    end
    return nothing
end

# Steady-state transverse magnetization right after the RF pulse, for
# flip angle α (radians), per M₀

# Spoiled gradient echo (Ernst)
ernst(α, TR, T₁) = sin(α) * (1 - exp(-TR / T₁)) / (1 - exp(-TR / T₁) * cos(α))

# Unbalanced SSFP, FID signal (FISP), with complete dephasing within a voxel
function fisp(α, TR, T₁, T₂)
    E₁, E₂ = exp(-TR / T₁), exp(-TR / T₂)
    p = 1 - E₁ * cos(α) - E₂^2 * (E₁ - cos(α))
    q = E₂ * (1 - E₁) * (1 + cos(α))
    return tan(α / 2) * (1 - (E₁ - cos(α)) * (1 - E₂^2) / sqrt(p^2 - q^2))
end

# Balanced SSFP with alternating RF phase, and off-resonance precession θ
# (radians) per TR
function bssfp(α, TR, T₁, T₂, θ)
    E₁, E₂ = exp(-TR / T₁), exp(-TR / T₂)
    θ += π # alternating RF phase
    return (1 - E₁) * sin(α) * sqrt(1 - 2E₂ * cos(θ) + E₂^2) /
           ((1 - E₁ * cos(α)) * (1 - E₂ * cos(θ)) - (E₁ - cos(α)) * (E₂ - cos(θ)) * E₂)
end

@testset "Validation against closed-form solutions" begin

    @testset "Spoiled gradient echo: Ernst formula (EPG operators)" begin
        TR, TE, nTR = 0.010, 0.004, 1500
        for (T₁, T₂, α, B₁) in [(0.9, 0.06, 20.0, 0.95), (0.3, 0.03, 70.0, 1.1), (1.5, 0.2, 10.0, 1.0)]
            sequence = IdealSpoiledGRE(α, nTR, TR, TE, Val(32))
            p = T₁T₂B₁(T₁, T₂, B₁)
            m, ∂m = simulate_derivatives_forward_sensitivity(sequence, p)
            closed_form(T₁, T₂, B₁) = ernst(deg2rad(α * B₁), TR, T₁) * exp(-TE / T₂)

            @test abs(m[end]) ≈ closed_form(T₁, T₂, B₁) rtol = 1e-12
            ∂closed_form = closed_form_derivatives(closed_form, T₁, T₂, B₁)
            for θ in (:T₁, :T₂, :B₁)
                @test ∂abs(m[end], ∂m[θ][end]) ≈ ∂closed_form[θ] rtol = 1e-7
            end
        end
    end

    @testset "Inversion recovery (FISP2D)" begin
        # One excitation, TI after an ideal inversion
        TE, α = 0.004, 25.0
        for (T₁, T₂, TI) in [(0.9, 0.06, 0.3), (1.5, 0.1, 1.0), (0.3, 0.03, 0.05)]
            sequence = FISP2D(complex.([α]), 0.010, TE, 32, TI)
            m = simulate_magnetization(CPU1(), sequence, StructVector([T₁T₂(T₁, T₂)]))[1]
            @test m ≈ -im * sin(deg2rad(α)) * (1 - 2exp(-TI / T₁)) * exp(-TE / T₂) rtol = 1e-12
        end
    end

    @testset "Unbalanced SSFP / FISP steady state (FISP3D, EPG model)" begin
        # The EPG truncation error is of the order of E₂^max_state, so TR/T₂
        # is chosen large enough for it to be negligible with 32 states. (More
        # states would allow longer T₂s, but compile much longer on the CPU.)
        TR, TE, nTR = 0.020, 0.004, 1500
        for (T₁, T₂, α, B₁) in [(1.0, 0.025, 30.0, 1.0), (0.5, 0.02, 60.0, 0.9), (1.5, 0.03, 15.0, 1.15)]
            sequence = FISP3D(complex.(fill(α, nTR)), TR, TE, Val(32), 0.0, 0.0, 1, false, false, 1, 0.0)
            p = T₁T₂B₁(T₁, T₂, B₁)
            m, ∂m = simulate_derivatives_forward_sensitivity(sequence, p)
            closed_form(T₁, T₂, B₁) = fisp(deg2rad(α * B₁), TR, T₁, T₂) * exp(-TE / T₂)

            @test abs(m[end]) ≈ closed_form(T₁, T₂, B₁) rtol = 1e-9
            ∂closed_form = closed_form_derivatives(closed_form, T₁, T₂, B₁)
            for θ in (:T₁, :T₂, :B₁)
                @test ∂abs(m[end], ∂m[θ][end]) ≈ ∂closed_form[θ] rtol = 1e-6
            end
        end
    end

    @testset "Balanced SSFP steady state (pSSFP3D, isochromat model)" begin
        # A hard pulse (a single RF sample of zero duration) with alternating
        # phase; echo time TR/2
        TR, nTR = 0.008, 3000
        for (T₁, T₂, α, B₁, B₀) in [(1.0, 0.1, 40.0, 1.0, 0.0), (0.8, 0.07, 60.0, 0.9, 12.0),
            (1.2, 0.15, 20.0, 1.1, -30.0)]
            RF_train = complex.(α .* (-1.0) .^ (0:nTR-1))
            sequence = pSSFP3D(RF_train, TR, SVector(deg2rad(1.0)), (ex=0.0, inv=0.0, pr=TR / 2))
            p = T₁T₂B₁B₀(T₁, T₂, B₁, B₀)
            m, ∂m = simulate_derivatives_forward_sensitivity(sequence, p)
            closed_form(T₁, T₂, B₁) = bssfp(deg2rad(α * B₁), TR, T₁, T₂, 2π * B₀ * TR) * exp(-TR / 2 / T₂)

            @test abs(m[end]) ≈ closed_form(T₁, T₂, B₁) rtol = 1e-9
            ∂closed_form = closed_form_derivatives(closed_form, T₁, T₂, B₁)
            for θ in (:T₁, :T₂, :B₁)
                @test ∂abs(m[end], ∂m[θ][end]) ≈ ∂closed_form[θ] rtol = 1e-6
            end
        end
    end

    @testset "Balanced SSFP steady state from waveforms (Generic3D)" begin
        # The same sequence, as RF and gradient waveforms: a 1 ns block pulse
        # (so relaxation during it is negligible), then two free precession
        # steps of TR/2, sampling in between on the last TR.
        TR, nTR, Δt_RF, α = 0.008, 2000, 1e-9, 40.0
        γ = 26753.0 # rad/s/G
        RF, Δt, sample = ComplexF64[], Float64[], Bool[]
        for n in 1:nTR
            append!(RF, [(-1)^n * deg2rad(α) / (γ * Δt_RF), 0, 0]) # G
            append!(Δt, [Δt_RF, (TR - Δt_RF) / 2, (TR - Δt_RF) / 2])
            append!(sample, [false, n == nTR, false])
        end
        sequence = Generic3D(RF, zeros(3, length(RF)), sample, Δt)

        for (T₁, T₂, B₀) in [(1.0, 0.1, 0.0), (0.8, 0.07, 7.0)]
            p = T₁T₂B₁B₀xyz(T₁, T₂, 1.0, B₀, 0.0, 0.0, 0.0)
            m = only(simulate_magnetization(CPU1(), sequence, StructVector([p])))
            @test hypot(m.x, m.y) ≈ bssfp(deg2rad(α), TR, T₁, T₂, 2π * B₀ * TR) * exp(-TR / 2 / T₂) rtol = 1e-7
        end
    end
end
