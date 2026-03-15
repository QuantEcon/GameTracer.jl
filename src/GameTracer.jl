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
export IPAResult, GNMResult

"""
    IPAResult

# Fields TBD

"""
# [TODO] TBD: Still under discussion
struct IPAResult{N}
    NE::NTuple{N, Vector{Float64}}
    nums_actions::NTuple{N, Int}
end

"""
    GNMResult

# Fields TBD

"""
# [TODO] TBD: Still under discussion
struct GNMResult{N}
    NEs::Vector{NTuple{N, Vector{Float64}}}
    nums_actions::NTuple{N, Int}
end



"""
    ipa!(ans::Vector{Float64}, g::NormalFormGame; kwargs...) -> ans

# Arguments

# Keyword Arguments

# Returns

- `ans`: The computed equilibrium strategy profile, modified in-place.

"""
function ipa!(
    ans::Vector{Float64},
    rng::AbstractRNG,
    g::NormalFormGame;
    ray::Union{Vector{Float64}, Nothing} = nothing,
    alpha::Float64 = 0.02,
    fuzz::Float64 = 1e-6,
)
    p = GAMPayoffVector(Float64, g)
    M = sum(p.nums_actions)

    length(ans) == M || throw(ArgumentError(
        "Length of ans $(length(ans)) must be equal to total number of actions (M=$M)"
    ))

    if ray === nothing
        ray = rand(rng, M)
    end
    z_hat = ones(M)
    
    result = ipa(p.nums_actions, p.payoffs, ray, z_hat, alpha, fuzz)

    copyto!(ans, result)
    
    return ans
end

ipa!(ans::Vector{Float64}, g::NormalFormGame; kwargs...) = 
    ipa!(ans, Random.GLOBAL_RNG, g; kwargs...)


"""
    ipa_solve(g::NormalFormGame) -> IPAResult

# Arguments

# Keyword Arguments

# Returns

- IPAResult:

# References

"""
# [TODO] TBD: Still under discussion
function ipa_solve(
    rng::AbstractRNG,
    g::NormalFormGame;
    ray::Union{Vector{Float64}, Nothing} = nothing,
    alpha::Float64 = 0.02,
    fuzz::Float64 = 1e-6,
)
    p = GAMPayoffVector(Float64, g)
    M = sum(p.nums_actions)

    ans = Vector{Float64}(undef, M)
    
    ipa!(ans, rng, g; ray=ray, alpha=alpha, fuzz=fuzz)

    NE = _slice_actions(ans, p.nums_actions)
    
    return IPAResult(NE, p.nums_actions)
end

ipa_solve(g::NormalFormGame; kwargs...) = 
    ipa_solve(Random.GLOBAL_RNG, g; kwargs...)


"""
    gnm_solve(g::NormalFormGame) -> GNMResult

# Arguments

# Keyword Arguments

# Returns

- GNMResult: 

# References

"""
# [TODO] TBD: Still under discussion
function gnm_solve(
    rng::AbstractRNG,
    g::NormalFormGame;
    ray::Union{Vector{Float64}, Nothing} = nothing,
    steps::Integer = 100,
    fuzz::Float64 = 1e-6,
    lnmfreq::Integer = 3,
    lnmmax::Integer = 10,
    lambdamin::Float64 = -10.0,
    wobble::Bool = false,
    threshold::Float64 = 1e-2
)
    p = GAMPayoffVector(Float64, g)
    M = sum(p.nums_actions)

    if ray === nothing
        ray = rand(rng, M)
    end

    equilibria_flat = gnm(p.nums_actions, p.payoffs, ray,
                      steps, fuzz, lnmfreq, lnmmax, 
                      lambdamin, wobble, threshold)
    
    NEs = [_slice_actions(ans, p.nums_actions) for ans in equilibria_flat]
    
    return GNMResult(NEs, p.nums_actions)
end

gnm_solve(g::NormalFormGame; kwargs...) = 
    gnm_solve(Random.GLOBAL_RNG, g; kwargs...)

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
    num_eq = size(x, 2)
    out = Vector{NTuple{N,Vector{T}}}(undef, num_eq)
    @inbounds for j in 1:num_eq
        out[j] = _get_action_profile(@view(x[:, j]), nums_actions)
    end
    return out
end

end
