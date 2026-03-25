module GameTracer

using GameTheory: NormalFormGame, GAMPayoffVector
using Random
using gametracer_jll: libgametracer

# ------------------------------------------------------------------
# Public API & Result Types
# ------------------------------------------------------------------
export ipa_solve, gnm_solve


"""
    IPAResult

Result of [`ipa_solve`](@ref).

# Fields
- `NE::NTuple{N, Vector{Float64}}`: Tuple of computed Nash equilibrium mixed 
    actions.
- `ret_code::Int`: Return code from the underlying C shim.
"""
struct IPAResult{N}
    NE::NTuple{N, Vector{Float64}}
    ret_code::Int
end

"""
    GNMResult

Result of [`gnm_solve`](@ref).

# Fields
- `NEs::Vector{NTuple{N, Vector{Float64}}}`: Vector of computed Nash equilibrium 
    mixed actions.
- `ret_code::Int`: Return code from the underlying C shim.
"""
struct GNMResult{N}
    NEs::Vector{NTuple{N, Vector{Float64}}}
    ret_code::Int
end


"""
    ipa_solve([rng=Random.GLOBAL_RNG, ]g; kwargs...)

Compute one mixed-action Nash equilibrium of `g` with the iterated polymatrix 
approximation (IPA) algorithm (Govindan and Wilson, 2004).

# Arguments
- `rng::AbstractRNG`: Random number generator used when the default `ray` is 
    used.
- `g::NormalFormGame`: A `NormalFormGame` instance with `N >= 2` players.

# Keyword Arguments
- `ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions))`: 
    Perturbation ray. Its length must equal `sum(g.nums_actions)`.
- `z_init::AbstractVector{<:Real} = ones(sum(g.nums_actions))`: 
    Initial point for the iteration. Its length must equal `sum(g.nums_actions)`.
- `alpha::Real = 0.02`: Step size parameter. Must satisfy `0 < alpha < 1`.
- `fuzz::Real = 1e-6`: Cutoff accuracy for the computed equilibrium.

# Returns
- `res::IPAResult{N}`: Result object containing information about 
    the computed equilibrium.
    - `res.NE`: Tuple of computed Nash equilibrium mixed actions.
    - `res.ret_code`: Return code from the underlying C shim.

# Examples

Consider the following 3x2 game from von Stengel (2007):

```julia
julia> using GameTheory, GameTracer, Random

julia> player1 = Player([3 3; 2 5; 0 6]);  

julia> player2 = Player([3 2 3; 2 6 1]);

julia> g = NormalFormGame(player1, player2);

julia> ray = [0.87, 0.04, 0.81, 0.97, 0.17];

julia> res = ipa_solve(g; ray=ray);

julia> println(res.ret_code > 0)
true

julia> println(length(res.NE))
2

julia> println(is_nash(g, res.NE; tol=1e-6))
true
```

# References
- S. Govindan and R. Wilson, "Computing Nash equilibria by iterated
  polymatrix approximation," Journal of Economic Dynamics and Control, 28 (2004),
  1229-1241.
"""
function ipa_solve(
    rng::AbstractRNG,
    g::NormalFormGame{N};
    ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions)),
    z_init::AbstractVector{<:Real} = ones(sum(g.nums_actions)),
    alpha::Real = 0.02,
    fuzz::Real = 1e-6,
) where {N}
    M = sum(g.nums_actions)

    length(ray) == M ||
        throw(ArgumentError("length(ray) must equal sum(g.nums_actions)"))
    length(z_init) == M || 
        throw(ArgumentError("length(z_init) must equal sum(g.nums_actions)"))
    0 < alpha < 1 || 
        throw(ArgumentError("alpha must satisfy 0 < alpha < 1"))
    
    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    z_init = Vector{Cdouble}(z_init)  # Copy
    out = Vector{Cdouble}(undef, M)
    out, ret_code = ipa!(
        N, actions, p.payoffs, ray, z_init, Cdouble(alpha), Cdouble(fuzz), out
    )

    NE = _get_action_profile(out, g.nums_actions)

    return IPAResult(NE, Int(ret_code))
end

ipa_solve(g::NormalFormGame; kwargs...) = 
    ipa_solve(Random.GLOBAL_RNG, g; kwargs...)

function ipa_solve(rng::AbstractRNG, g::NormalFormGame{1}; kwargs...)
    throw(ArgumentError("not implemented for 1-player games"))
end


"""
    gnm_solve([rng=Random.GLOBAL_RNG, ]g; kwargs...)

Compute mixed-action Nash equilibria of `g` with the global Newton method (GNM) 
algorithm (Govindan and Wilson, 2003). 

# Arguments
- `rng::AbstractRNG`: Random number generator used when the default `ray` is 
    used.
- `g::NormalFormGame`: A `NormalFormGame` instance with `N >= 2` players.

# Keyword Arguments
- `ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions))`: 
    Perturbation ray. Its length must equal `sum(g.nums_actions)`.
- `steps::Integer = 100`: Maximum number of steps.
- `fuzz::Real = 1e-12`: Cutoff value for a variety of things.
- `lnmfreq::Integer = 3`: Frequency parameter. A Local Newton Method subroutine 
    will be run every LNMFreq steps to decrease accumulated errors.
- `lnmmax::Integer = 10`: Maximum number of iterations within the LNM algorithm.
- `lambdamin::Real = -10.0`: Minimum lambda value. The equilibrium search 
    terminates if lambda falls below this value. Must be negative.
- `wobble::Bool = false`: Whether to use "wobbles" of the perturbation vector to
     remove accumulated errors. This removes the theoretical guarantee of 
     convergence, but in practice may help keep GNM on the path 
- `threshold::Real = 1e-2`: The equilibrium error tolerance for doing a wobble. 
    If wobbles are disabled, the GNM algorithm terminates if the error reaches 
    this threshold.

# Returns
- `res::GNMResult{N}`: Result object containing information about 
    the computed equilibria.
    - `res.NEs`: Vector of computed Nash equilibrium mixed actions.
    - `res.ret_code`: Return code from the underlying C shim.

# Examples

Consider the following 2x2x2 game with 9 Nash equilibria from McKelvey and 
    McLennan (1996):

```julia
julia> using GameTheory, GameTracer, Random

julia> g = NormalFormGame((2, 2, 2));

julia> g[1, 1, 1] = 9, 8, 12;

julia> g[2, 2, 1] = 9, 8, 2;

julia> g[1, 2, 2] = 3, 4, 6;

julia> g[2, 1, 2] = 3, 4, 4;

julia> ray = [0.87, 0.04, 0.81, 0.97, 0.17, 0.04]

julia> res = gnm_solve(g; ray=ray);

julia> println(res.ret_code)
9

julia> println(length(res.NEs))
9

julia> println(all(NE -> is_nash(g, NE), res.NEs))
true
```

# References
- S. Govindan and R. Wilson, "A global Newton method to compute Nash 
    equilibria," Journal of Economic Theory, 110 (2003), 65-86.
"""
function gnm_solve(
    rng::AbstractRNG,
    g::NormalFormGame{N};
    ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions)),
    steps::Integer = 100,
    fuzz::Real = 1e-12,
    lnmfreq::Integer = 3,
    lnmmax::Integer = 10,
    lambdamin::Real = -10.0,
    wobble::Bool = false,
    threshold::Real = 1e-2
) where {N}
    M = sum(g.nums_actions)

    length(ray) == M ||
        throw(ArgumentError("length(ray) must equal sum(g.nums_actions)"))
    lambdamin < 0 || 
        throw(ArgumentError("lambdamin must be negative"))

    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    answers, ret_code = gnm(
        N, actions, p.payoffs, ray,
        steps, Cdouble(fuzz), lnmfreq, lnmmax,
        Cdouble(lambdamin), wobble, Cdouble(threshold)
    )

    NEs = _get_action_profiles(answers, g.nums_actions)
    
    return GNMResult(NEs, Int(ret_code))
end

gnm_solve(g::NormalFormGame; kwargs...) = 
    gnm_solve(Random.GLOBAL_RNG, g; kwargs...)

function gnm_solve(rng::AbstractRNG, g::NormalFormGame{1}; kwargs...)
    throw(ArgumentError("not implemented for 1-player games"))
end


# ------------------------------------------------------------------
# Private API (C ABI wrappers)
# ------------------------------------------------------------------

function ipa!(
    N::Integer,
    actions::Vector{Cint},
    payoffs::Vector{Cdouble},
    ray::Vector{Cdouble},
    zh::Vector{Cdouble},
    alpha::Cdouble,
    fuzz::Cdouble,
    out::Vector{Cdouble}
)
    ret = ccall(
        (:ipa, libgametracer), Cint,
        (Cint, Ptr{Cint}, Ptr{Cdouble},
         Ptr{Cdouble}, Ptr{Cdouble},
         Cdouble, Cdouble,
         Ptr{Cdouble}),
        N, actions, payoffs,
        ray, zh,
        alpha, fuzz,
        out
    )

    ret <= 0 && error("IPA failed (ret = $ret)")

    return (out, ret)
end

function gnm(
    N::Integer,
    actions::Vector{Cint},
    payoffs::Vector{Cdouble},
    ray::Vector{Cdouble},
    steps::Integer,
    fuzz::Cdouble,
    lnmfreq::Integer,
    lnmmax::Integer,
    lambdamin::Cdouble,
    wobble::Integer,
    threshold::Cdouble,
)
    M = sum(actions)
    answers_ref = Ref{Ptr{Cdouble}}(C_NULL)

    ret = ccall(
        (:gnm, libgametracer), Cint,
        (Cint, Ptr{Cint}, Ptr{Cdouble},
         Ptr{Cdouble}, Ref{Ptr{Cdouble}},
         Cint, Cdouble, Cint, Cint, Cdouble, Cint, Cdouble),
        N, actions, payoffs,
        ray, answers_ref,
        steps, fuzz, lnmfreq, lnmmax, lambdamin, wobble, threshold
    )

    ret < 0 && error("GNM failed (ret = $ret)")

    # ret == 0: 0 equilibria, answers == NULL
    ret == 0 && return (Matrix{Cdouble}(undef, M, 0), ret)

    # ret > 0: num_eq equilibria, answers is malloc'd buffer
    ptr = answers_ref[]
    ptr != C_NULL || error("GNM returned ret=$ret but answers pointer was NULL")

    answers = try
        answers_view = unsafe_wrap(Array, ptr, (M, Int(ret)); own=false)
        copy(answers_view)
    finally
        ccall((:gametracer_free, libgametracer), Cvoid, (Ptr{Cvoid},), ptr)
    end

    return (answers, ret)
end

function _get_action_profile(x::AbstractVector{T},
                             nums_actions::NTuple{N,Integer}) where {N,T}
    out = ntuple(i -> Vector{T}(undef, nums_actions[i]), Val(N))
    ind = 1
    @inbounds for i in 1:N
        len = nums_actions[i]
        copyto!(out[i], 1, x, ind, len)
        ind += len
    end
    return out
end

function _get_action_profiles(x::AbstractMatrix{T},
                              nums_actions::NTuple{N,Integer}) where {N,T}
    num_NEs = size(x, 2)
    out = Vector{NTuple{N,Vector{T}}}(undef, num_NEs)
    @inbounds for j in 1:num_NEs
        out[j] = _get_action_profile(@view(x[:, j]), nums_actions)
    end
    return out
end

end
