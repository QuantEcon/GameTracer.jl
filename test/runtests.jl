using GameTracer
using GameTheory
using Random
using Test

# Whether `x` is a valid mixed action profile of a game with `nums_actions`
function is_mixed_action_profile(x, nums_actions; tol=1e-12)
    length(x) == length(nums_actions) || return false
    for (xi, n) in zip(x, nums_actions)
        length(xi) == n || return false
        all(>=(-tol), xi) || return false
        isapprox(sum(xi), 1, atol=tol) || return false
    end
    return true
end

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
            @test res.converged
            @test res.ret_code == 1
            @test res.num_iter >= 1
            @test res.max_iter == 100000

            # Heuristic; no epsilon-optimality guarantee derived in the paper
            tol = fuzz_default * payoff_max
            @test is_nash(g, res.NE, tol=tol)

            fuzz = 1e-8
            res = @inferred ipa_solve(rng, g, fuzz=fuzz)
            tol = fuzz * payoff_max
            @test is_nash(g, res.NE, tol=tol)
        end

        @testset "IPAResult.ray" begin
            g = gs[2]
            ray = zeros(sum(g.nums_actions))
            ray[[cumsum(g.nums_actions)...]] .= 1
            res = @inferred ipa_solve(g, ray=ray)
            @test res.ray == ray
        end
    end

    @testset "gnm_solve" begin
        seed = 1234
        rng = MersenneTwister(seed)
        for g in gs
            res = @inferred gnm_solve(rng, g)
            @test length(res.NEs) == res.ret_code
            @test res.num_iter >= 1
            @test res.max_iter == 5000
            for NE in res.NEs
                @test is_nash(g, NE,)
            end
        end

        @testset "GNMResult.ray" begin
            g = gs[2]
            ray = zeros(sum(g.nums_actions))
            ray[[cumsum(g.nums_actions)...]] .= 1
            res = @inferred gnm_solve(g, ray=ray)
            @test res.ray == ray
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
        @test_throws ArgumentError ipa_solve(rng, g, max_iter=0)
        @test_throws ArgumentError ipa_solve(rng, g, max_pivots=0)
        @test_throws ArgumentError ipa_solve(rng, g, max_iter=typemax(Int))
        @test_throws ArgumentError ipa_solve(rng, g, max_pivots=typemax(Int))
    end

    @testset "ipa_solve iteration limits" begin
        # 2x2x2 game: needs more than 10 iterations with this ray
        g = gs[2]
        ray = [0.3, 0.7, 0.6, 0.4, 0.2, 0.8]
        res = ipa_solve(g, ray=ray)
        @test res.converged
        needed = res.num_iter
        @test needed > 10

        # Iteration limit reached: the last iterate is returned
        for cap in (1, 10)
            res = @inferred ipa_solve(g, ray=ray, max_iter=cap)
            @test !res.converged
            @test res.ret_code == 0
            @test res.num_iter == cap
            @test res.max_iter == cap
            @test is_mixed_action_profile(res.NE, g.nums_actions)
        end

        # Exactly enough iterations
        res = ipa_solve(g, ray=ray, max_iter=needed)
        @test res.converged
        @test res.num_iter == needed
        @test is_nash(g, res.NE, tol=1e-5)

        # Pivot limit: the solver gives up and returns the last iterate
        res = ipa_solve(g, ray=ray, max_pivots=2)
        @test !res.converged
        @test res.ret_code == 0
        @test 1 <= res.num_iter < needed
        @test is_mixed_action_profile(res.NE, g.nums_actions)

        # 3x2 game: the Lemke-Howson path has exactly 5 pivots
        g = gs[1]
        ray = [0.0, 0.0, 1.0, 0.0, 1.0]
        zh_init = [1/3, 1/3, 1/3, 1/2, 1/2]
        res = ipa_solve(g, ray=ray, zh_init=zh_init, max_pivots=4)
        @test !res.converged
        @test res.num_iter == 1
        @test is_mixed_action_profile(res.NE, g.nums_actions)
        res = ipa_solve(g, ray=ray, zh_init=zh_init, max_pivots=5)
        @test res.converged
        @test res.num_iter == 1
        @test is_nash(g, res.NE, tol=1e-6)

        # Giving up during the last allowed iteration also leaves
        # num_iter == max_iter
        res = ipa_solve(g, ray=ray, zh_init=zh_init, max_iter=1, max_pivots=4)
        @test !res.converged
        @test res.num_iter == 1
    end

    @testset "gnm_solve input validation" begin
        g = gs[1]
        seed = 1234
        rng = MersenneTwister(seed)
        M = sum(g.nums_actions)
        @test_throws ArgumentError gnm_solve(rng, g, ray=zeros(M - 1))
        @test_throws ArgumentError gnm_solve(rng, g, lambdamin=0.0)
        @test_throws ArgumentError gnm_solve(rng, g, max_iter=0)
        @test_throws ArgumentError gnm_solve(rng, g, max_iter=typemax(Int))
    end

    @testset "gnm_solve iteration limits" begin
        # 3x2 game: the three equilibria are found one by one along the path
        g = gs[1]
        ray = [0.0, 0.0, 1.0, 0.0, 1.0]
        res = gnm_solve(g, ray=ray)
        @test length(res.NEs) == 3
        needed = res.num_iter
        @test needed > 3

        prev = 0
        for cap in 1:needed-1
            res = @inferred gnm_solve(g, ray=ray, max_iter=cap)
            @test 0 <= res.ret_code <= 3
            @test res.ret_code >= prev  # Found in path order
            @test res.num_iter == cap
            @test res.max_iter == cap
            for NE in res.NEs
                @test is_nash(g, NE)
            end
            prev = res.ret_code
        end
        @test prev < 3  # The last crossing is needed for the third one

        res = gnm_solve(g, ray=ray, max_iter=needed)
        @test length(res.NEs) == 3
        @test res.num_iter == needed

        # 2x2x2 game (nonlinear path within a cell)
        g = gs[2]
        ray = [0.3, 0.7, 0.6, 0.4, 0.2, 0.8]
        res_full = gnm_solve(g, ray=ray)
        @test length(res_full.NEs) > 1
        needed = res_full.num_iter
        @test needed > 1
        res = gnm_solve(g, ray=ray, max_iter=1)
        @test length(res.NEs) < length(res_full.NEs)
        @test res.num_iter == 1
        for NE in res.NEs
            @test is_nash(g, NE)
        end
        res = gnm_solve(g, ray=ray, max_iter=needed)
        @test length(res.NEs) == length(res_full.NEs)
        @test res.num_iter == needed
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
