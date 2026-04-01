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

Struct that stores the output of the IPA solver.

# Fields

- `NE::NTuple{N, Vector{Float64}}`: Tuple of computed Nash equilibrium mixed 
  actions.
- `ret_code::Int`: Return code from the underlying C shim: `1` on success.
"""
struct IPAResult{N}
    NE::NTuple{N, Vector{Float64}}
    ret_code::Int
end

"""
    GNMResult

Struct that stores the output of the GNM solver.

# Fields

- `NEs::Vector{NTuple{N, Vector{Float64}}}`: Vector of tuples of computed Nash 
  equilibrium mixed actions.
- `ret_code::Int`: Return code from the underlying C shim: the number of
  equilibria found.
"""
struct GNMResult{N}
    NEs::Vector{NTuple{N, Vector{Float64}}}
    ret_code::Int
end


@doc raw"""
    ipa_solve(g; kwargs...)
    ipa_solve(rng, g; kwargs...)

Compute one mixed-action approximate Nash equilibrium of `g` with the iterated
polymatrix approximation (IPA) algorithm (Govindan and Wilson, 2004).

# Arguments

- `rng::AbstractRNG = Random.GLOBAL_RNG`: Random number generator used to
  randomly generate the default `ray`.
- `g::GameTheory.NormalFormGame`: `NormalFormGame` instance with `N >= 2`
  players.

# Keywords

- `ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions))`: Perturbation
  ray for the IPA homotopy. Its length must equal `sum(g.nums_actions)`, and
  entries are interpreted in player-concatenated order. Different rays may lead
  to different equilibria.
- `z_init::AbstractVector{<:Real} = ones(sum(g.nums_actions))`: Initial
  condition for ``\hat{z}``. Its length must equal `sum(g.nums_actions)`.
  IPA starts with the projection of `z_init` onto the mixed-action simplex
  product.
- `alpha::Real = 0.02`: Step size fraction of an update of ``\hat{z}``. Must
  satisfy `0 < alpha < 1`.
- `fuzz::Real = 1e-6`: Stopping tolerance for the computed equilibrium.

# Returns

- `res::IPAResult`: Result object containing information about the computed
  equilibrium. `res.NE` contains an `N` tuple of mixed actions, one for each
  player.

# Examples

Consider the following 2x2x2 game from McKelvey and McLennan (1996), which is
known to have 9 Nash equilibria:

```julia
julia> using GameTheory, GameTracer, Random

julia> Base.active_repl.options.iocontext[:compact] = true;  # Reduce digits to display

julia> rng = MersenneTwister(0);

julia> g = NormalFormGame((2, 2, 2));

julia> g[1, 1, 1] = [9, 8, 12];

julia> g[2, 2, 1] = [9, 8, 2];

julia> g[1, 2, 2] = [3, 4, 6];

julia> g[2, 1, 2] = [3, 4, 4];

julia> println(g)
2×2×2 NormalFormGame{3, Float64}:
[:, :, 1] =
 [9.0, 8.0, 12.0]  [0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0]   [9.0, 8.0, 2.0]

[:, :, 2] =
 [0.0, 0.0, 0.0]  [3.0, 4.0, 6.0]
 [3.0, 4.0, 4.0]  [0.0, 0.0, 0.0]

julia> res = ipa_solve(rng, g);

julia> res.NE
([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])

julia> is_nash(g, res.NE)
true
```

Calls with different rays generally yield different equilibria (here, `rng`
generates a different ray on each call as its state advances):

```julia
julia> res = ipa_solve(rng, g);

julia> res.NE
([0.25, 0.75], [0.5, 0.5], [0.333334, 0.666666])

julia> is_nash(g, res.NE; tol=1e-5)  # Relax tolerance
true
```

# References

- S. Govindan and R. Wilson, "Computing Nash equilibria by iterated polymatrix
  approximation," Journal of Economic Dynamics and Control, 28 (2004),
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


@doc raw"""
    gnm_solve(g; kwargs...)
    gnm_solve(rng, g; kwargs...)

Compute multiple mixed-action Nash equilibria of `g` with the global Newton
method (GNM) algorithm (Govindan and Wilson, 2003).

# Arguments

- `rng::AbstractRNG = Random.GLOBAL_RNG`: Random number generator used to
  randomly generate the default `ray`.
- `g::GameTheory.NormalFormGame`: `NormalFormGame` instance with `N >= 2`
  players.

# Keywords

- `ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions))`: Perturbation
  ray for the GNM homotopy. Its length must equal `sum(g.nums_actions)`, and
  entries are interpreted in player-concatenated order. Different rays may lead
  to different sets of equilibria.
- `steps::Integer = 100`: Number of steps to take within a support cell.
- `fuzz::Real = 1e-12`: Numerical zero threshold used throughout GNM.
- `lnmfreq::Integer = 3`: Frequency of local newton method (LNM) corrections.
  An LNM subroutine will be run every `lnmfreq` steps to decrease accumulated
  errors.
- `lnmmax::Integer = 10`: Maximum number of iterations within the LNM
  subroutine.
- `lambdamin::Real = -10.0`: Minimum value for the continuation parameter
  ``\lambda``. The equilibrium search terminates if ``\lambda`` falls below
  this value. Must be negative.
- `wobble::Bool = false`: Whether to use "wobbles" of the perturbation vector
  to remove accumulated errors. This removes the theoretical guarantee of
  convergence, but in practice may help keep GNM on the path.
- `threshold::Real = 1e-2`: Error threshold used to trigger a wobble. If
  `wobble == false`, the GNM algorithm terminates if the error reaches this
  threshold.

# Returns

- `res::GNMResult`: Result object containing information about the computed
  equilibria. `res.NEs` contains a vector of `N` tuples of mixed actions, one
  for each equilibrium.

# Examples

Consider the following 2x2x2 game from McKelvey and McLennan (1996), which is
known to have 9 Nash equilibria:

```julia
julia> using GameTheory, GameTracer, Random

julia> Base.active_repl.options.iocontext[:compact] = true;  # Reduce digits to display

julia> rng = MersenneTwister(50);

julia> g = NormalFormGame((2, 2, 2));

julia> g[1, 1, 1] = [9, 8, 12];

julia> g[2, 2, 1] = [9, 8, 2];

julia> g[1, 2, 2] = [3, 4, 6];

julia> g[2, 1, 2] = [3, 4, 4];

julia> println(g)
2×2×2 NormalFormGame{3, Float64}:
[:, :, 1] =
 [9.0, 8.0, 12.0]  [0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0]   [9.0, 8.0, 2.0]

[:, :, 2] =
 [0.0, 0.0, 0.0]  [3.0, 4.0, 6.0]
 [3.0, 4.0, 4.0]  [0.0, 0.0, 0.0]

julia> res = gnm_solve(rng, g);

julia> res.NEs
2-element Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}:
 ([0.0, 1.0], [0.0, 1.0], [1.0, 0.0])
 ([0.0, 1.0], [0.333333, 0.666667], [0.333333, 0.666667])
```

Calls with different rays generally yield different sets of equilibria (here,
`rng` generates a different ray on each call as its state advances):

```julia
julia> res = gnm_solve(rng, g);

julia> res.NEs
9-element Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}:
 ([1.0, 0.0], [0.0, 1.0], [0.0, 1.0])
 ([0.5, 0.5], [0.333333, 0.666667], [0.25, 0.75])
 ([0.0, 1.0], [0.0, 1.0], [1.0, 0.0])
 ([0.0, 1.0], [0.333333, 0.666667], [0.333333, 0.666667])
 ([0.25, 0.75], [0.5, 0.5], [0.333333, 0.666667])
 ([0.5, 0.5], [0.5, 0.5], [1.0, 0.0])
 ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
 ([0.25, 0.75], [1.0, 0.0], [0.25, 0.75])
 ([0.0, 1.0], [1.0, 0.0], [0.0, 1.0])

julia> all([is_nash(g, NE) for NE in res.NEs])
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
