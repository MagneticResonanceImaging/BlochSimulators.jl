# Deliberately flawed variants of a small spoiled gradient echo sequence, to
# check that check_sequence catches each flaw. `flaw` selects which one.
struct FlawedGRE{T,Ns,flaw,U<:AbstractVector{T}} <: EPGSimulator{T,Ns}
    RF_train::U
    TR::T
    TE::T
    max_state::Val{Ns}
    flaw::Val{flaw}
end
BlochSimulators.@functor FlawedGRE
BlochSimulators.@adapt_structure FlawedGRE

BlochSimulators.output_size(sequence::FlawedGRE) = length(sequence.RF_train)
BlochSimulators.output_eltype(sequence::FlawedGRE{T}) where {T} = Complex{T}

function BlochSimulators.simulate_magnetization!(magnetization, sequence::FlawedGRE{T,Ns,flaw}, Ω, p::AbstractTissueProperties) where {T,Ns,flaw}
    if flaw == :direct_relaxation
        E₁ᵀᴱ, E₂ᵀᴱ = exp(-sequence.TE / p.T₁), exp(-sequence.TE / p.T₂)
    else
        E₁ᵀᴱ, E₂ᵀᴱ = BlochSimulators.E₁(Ω, sequence.TE, p.T₁), BlochSimulators.E₂(Ω, sequence.TE, p.T₂)
    end
    E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ = BlochSimulators.E₁(Ω, sequence.TR - sequence.TE, p.T₁), BlochSimulators.E₂(Ω, sequence.TR - sequence.TE, p.T₂)
    flaw == :allocating && (sequence = deepcopy(sequence))

    BlochSimulators.initial_conditions!(Ω)
    for (TR, RF) in enumerate(sequence.RF_train)
        BlochSimulators.excite!(Ω, RF, p)
        BlochSimulators.decay!(Ω, E₁ᵀᴱ, E₂ᵀᴱ)
        BlochSimulators.regrowth!(Ω, E₁ᵀᴱ)
        flaw == :no_sampling || BlochSimulators.sample_transverse!(magnetization, TR, Ω)
        BlochSimulators.decay!(Ω, E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ)
        BlochSimulators.regrowth!(Ω, E₁ᵀᴿ⁻ᵀᴱ)
        BlochSimulators.dephasing!(Ω)
    end
    return nothing
end

FlawedGRE(flaw) = FlawedGRE(collect(range(10.0, 50.0, length=30)), 0.01, 0.004, Val(32), Val(flaw))

# Without @adapt_structure. (A test without @functor is not possible:
# since Functors 0.5, every struct is a functor.) The sequence is nested,
# which also checks that the GPU detection looks inside fields.
struct GRENoAdapt{T,Ns,S<:FlawedGRE{T,Ns}} <: EPGSimulator{T,Ns}
    sequence::S
end
BlochSimulators.@functor GRENoAdapt
BlochSimulators.output_size(s::GRENoAdapt) = BlochSimulators.output_size(s.sequence)
BlochSimulators.output_eltype(s::GRENoAdapt) = BlochSimulators.output_eltype(s.sequence)
BlochSimulators.simulate_magnetization!(m, s::GRENoAdapt, Ω, p::AbstractTissueProperties) =
    BlochSimulators.simulate_magnetization!(m, s.sequence, Ω, p)

# Without simulate_magnetization!
struct GRENoSimulation{T,Ns} <: EPGSimulator{T,Ns} end
BlochSimulators.output_size(::GRENoSimulation) = 10
BlochSimulators.output_eltype(::GRENoSimulation{T}) where {T} = Complex{T}

@testset "check_sequence" begin

    T₁, T₂, B₁ = [0.8, 1.2], [0.08, 0.1], [0.9, 1.1]
    parameters = @parameters T₁ T₂ B₁

    function run_check(sequence)
        io = IOBuffer()
        passed = check_sequence(sequence, parameters; io)
        return passed, String(take!(io))
    end
    failed(output, check) = occursin("✗ $check", output)

    passed, output = run_check(FlawedGRE(:none))
    @test passed
    @test !occursin("✗", output)

    # a single set of tissue properties works too
    @test check_sequence(FlawedGRE(:none), first(parameters); io=devnull)

    passed, output = run_check(GRENoSimulation{Float64,32}())
    @test !passed && failed(output, "output_size, output_eltype and simulate_magnetization!")
    @test occursin("Missing methods for GRENoSimulation: simulate_magnetization!", output)

    passed, output = run_check(FlawedGRE(:no_sampling))
    @test !passed && failed(output, "simulates on the CPU")

    passed, output = run_check(FlawedGRE(:allocating))
    @test !passed && failed(output, "simulate_magnetization! does not allocate")

    passed, output = run_check(FlawedGRE(:direct_relaxation))
    @test !passed && failed(output, "simulate_derivatives_forward_sensitivity agrees")
    @test occursin("E₁(state, Δt, T₁)", output)

    if CUDA.functional()
        passed, output = run_check(GRENoAdapt(FlawedGRE(:none)))
        @test !passed && failed(output, "runs on the GPU")
        @test occursin("@adapt_structure GRENoAdapt", output)
    end
end
