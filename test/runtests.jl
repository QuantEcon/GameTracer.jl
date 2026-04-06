using GameTracer
using GameTheory
using Random
using Test

@testset "GameTracer.jl" begin
    gs = []

    g = NormalFormGame(Player([3 3; 2 5; 0 6]),
                       Player([3 2 3; 2 6 1]))
    push!(gs, g)

    g = NormalFormGame((2, 2, 2))
    g[1, 1, 1] = 9, 8, 12
    g[2, 2, 1] = 9, 8, 2
    g[1, 2, 2] = 3, 4, 6
    g[2, 1, 2] = 3, 4, 4
    push!(gs, g)

    @testset "ipa_solve" begin
        seed = 1234
        rng = MersenneTwister(seed)
        fuzz_default = 1e-6
        for g in gs
            payoff_max = maximum(
                maximum(player.payoff_array) for player in g.players
            )

            res = @inferred ipa_solve(rng, g)
            @test res.ret_code > 0

            # Heuristic; no epsilon-optimality guarantee derived in the paper
            tol = fuzz_default * payoff_max
            @test is_nash(g, res.NE, tol=tol)

            fuzz = 1e-8
            res = @inferred ipa_solve(rng, g, fuzz=fuzz)
            tol = fuzz * payoff_max
            @test is_nash(g, res.NE, tol=tol)
        end
    end

    @testset "gnm_solve" begin
        seed = 1234
        rng = MersenneTwister(seed)
        for g in gs
            res = @inferred gnm_solve(rng, g)
            @test length(res.NEs) == res.ret_code
            for NE in res.NEs
                @test is_nash(g, NE,)
            end
        end
    end

    @testset "ipa_solve input validation" begin
        g = gs[1]
        seed = 1234
        rng = MersenneTwister(seed)
        M = sum(g.nums_actions)
        @test_throws ArgumentError ipa_solve(rng, g, ray=zeros(M - 1))
        @test_throws ArgumentError ipa_solve(rng, g, zh_init=ones(M - 1))
        @test_throws ArgumentError ipa_solve(rng, g, alpha=-0.1)
        @test_throws ArgumentError ipa_solve(rng, g, alpha=1.5)
    end

    @testset "gnm_solve input validation" begin
        g = gs[1]
        seed = 1234
        rng = MersenneTwister(seed)
        M = sum(g.nums_actions)
        @test_throws ArgumentError gnm_solve(rng, g, ray=zeros(M - 1))
        @test_throws ArgumentError gnm_solve(rng, g, lambdamin=0.0)
    end

    @testset "1-player game" begin
        g = NormalFormGame([[1], [2], [3]])
        @test_throws ArgumentError ipa_solve(g)
        @test_throws ArgumentError gnm_solve(g)
    end

    @testset "action-profile helpers" begin
        num_actions = (2, 3)
        x = [0.2, 0.8, 0.5, 0.3, 0.2]
        @test GameTracer._get_action_profile(x, num_actions) ==
              ([0.2, 0.8], [0.5, 0.3, 0.2])
        
        X = [
            0.2 0.5
            0.8 0.3 
            0.5 0.1
            0.3 0.2
            0.2 0.7
        ]
        @test GameTracer._get_action_profiles(X, num_actions) == [
            ([0.2, 0.8], [0.5, 0.3, 0.2]),
            ([0.5, 0.3], [0.1, 0.2, 0.7])
        ]
    end
end
