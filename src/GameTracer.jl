module GameTracer

using GameTheory: NormalFormGame, GAMPayoffVector
using Random

# ------------------------------------------------------------------
# Library Path
# ------------------------------------------------------------------
# TODO: Remove hardcoded dylib path after Phase B
# Temporary: Using local dylib until Phase B is complete
const libgametracer = "/tmp/gametracer_prefix/lib/libgametracer.dylib"

# ------------------------------------------------------------------
# Public API & Result Types
# ------------------------------------------------------------------
export ipa_solve, gnm_solve


"""
    IPAResult

# Fields
- `NE::NTuple{N, Vector{Float64}}`: one Nash equilibrium in mixed strategies.
"""
struct IPAResult{N}
    NE::NTuple{N, Vector{Float64}}
end

"""
    GNMResult

# Fields
- `NEs::Vector{NTuple{N, Vector{Float64}}}`: equilibria found by GNM.
- `num_NEs::Int`: number of equilibria found.
"""
struct GNMResult{N}
    NEs::Vector{NTuple{N, Vector{Float64}}}
    num_NEs::Int
end


"""
    ipa_solve(g::NormalFormGame) -> IPAResult

Compute one Nash equilibrium using the IPA algorithm.

# Arguments
- `g::NormalFormGame`: the game to solve (must have 2 or more players)

# Keyword Arguments
- `ray::AbstractVector{<:Real}`: 
    initial ray for IPA (default: random vector of length sum of g.nums_actions)
- `z_init::AbstractVector{<:Real}`: 
    initial z_init vector for IPA (default: vector of ones of length sum of g.nums_actions)
- `alpha::Real`: step size parameter for IPA (default: 0.02)
- `fuzz::Real`: convergence threshold for IPA (default: 1e-6)

# Returns
- `IPAResult`: one equilibrium found by IPA.

# References
- S. Govindan and R. Wilson (2004), "Computing Nash equilibria by iterated
  polymatrix approximation", Journal of Economic Dynamics and Control 28,
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
        throw(ArgumentError("length(ray) must be equal to sum(g.nums_actions)"))
    length(z_init) == M || 
        throw(ArgumentError("length(z_init) must be equal to sum(g.nums_actions)"))
    0 < alpha < 1 || 
        throw(ArgumentError("alpha must satisfy 0 < alpha < 1"))
    
    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    z_init = Vector{Cdouble}(z_init)  # Copy
    out = Vector{Cdouble}(undef, M)
    out, ret = ipa!(
        N, actions, p.payoffs, ray, z_init, Cdouble(alpha), Cdouble(fuzz), out
    )

    NE = _get_action_profile(out, g.nums_actions)

    return IPAResult(NE)
end

ipa_solve(g::NormalFormGame; kwargs...) = 
    ipa_solve(Random.GLOBAL_RNG, g; kwargs...)

function ipa_solve(rng::AbstractRNG, g::NormalFormGame{1}; kwargs...)
    throw(ArgumentError("not implemented for 1-player games"))
end


"""
    gnm_solve(g::NormalFormGame) -> GNMResult

Compute Nash equilibria using the GNM algorithm.

# Arguments
- `g::NormalFormGame`: the game to solve (must have 2 or more players)

# Keyword Arguments
- `ray::AbstractVector{<:Real}`: 
    initial ray for GNM (default: random vector of length sum of g.nums_actions)
- `steps::Integer`: maximum number of steps for GNM (default: 100)
- `fuzz::Real`: convergence threshold for GNM (default: 1e-12)
- `lnmfreq::Integer`: frequency of LNM calls in GNM (default: 3)
- `lnmmax::Integer`: maximum number of LNM calls in GNM (default: 10)
- `lambdamin::Real`: minimum lambda value for LNM in GNM (default: -10.0)
- `wobble::Bool`: whether to use wobbling in GNM (default: false)
- `threshold::Real`: threshold for wobbling in GNM (default: 1e-2)

# Returns
- `GNMResult`: equilibria found by GNM.

# References
- S. Govindan and R. Wilson (2003), "A global Newton method to compute Nash
  equilibria", Journal of Economic Theory 110, 65-86.
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
        throw(ArgumentError("length(ray) must be sum(g.nums_actions)"))
    lambdamin < 0 || 
        throw(ArgumentError("lambdamin must be a negative finite value"))

    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    answers, ret = gnm(
        N, actions, p.payoffs, ray,
        steps, Cdouble(fuzz), lnmfreq, lnmmax,
        Cdouble(lambdamin), wobble, Cdouble(threshold)
    )

    NEs = _get_action_profiles(answers, g.nums_actions)
    
    return GNMResult(NEs, Int(ret))
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
