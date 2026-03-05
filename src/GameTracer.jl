module GameTracer

using GameTheory: NormalFormGame, GAMPayoffVector

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
    ipa_solve(g::NormalFormGame) -> IPAResult

# Arguements

# Keyword Arguments

# Returns

- IPAResult:

# References

"""
# [TODO] TBD: Still under discussion
function ipa_solve(
    g::NormalFormGame;
    ray::Union{Vector{Float64}, Nothing} = nothing,
    alpha::Float64 = 0.02,
    fuzz::Float64 = 1e-6,
)
    p = GAMPayoffVector(Float64, g)
    M = sum(p.nums_actions)

    if ray === nothing
        ray = rand(M)
    end
    zh = ones(M)
    
    ans_flat = _ipa(p.nums_actions, p.payoffs, ray, zh, alpha, fuzz)

    NE = _slice_actions(ans_flat, p.nums_actions)
    
    return IPAResult(NE, p.nums_actions)
end


"""
    gnm_solve(g::NormalFormGame) -> GNMResult

# Arguements

# Keyword Arguments

# Returns

- GNMResult: 

# References

"""
# [TODO] TBD: Still under discussion
function gnm_solve(
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
        ray = rand(M)
    end

    equilibria_flat = _gnm(p.nums_actions, p.payoffs, ray,
                      steps, fuzz, lnmfreq, lnmmax, 
                      lambdamin, wobble, threshold)
    
    NEs = [_slice_actions(ans, p.nums_actions) for ans in equilibria_flat]
    
    return GNMResult(NEs, p.nums_actions)
end

# ------------------------------------------------------------------
# Private API (C ABI wrappers)
# ------------------------------------------------------------------

# Slice a flat Vector{Float64} of length M into per-player NTuple
function _slice_actions(
    flat::Vector{Float64},
    nums_actions::NTuple{N,Int}
) where N
    offsets = cumsum((0, nums_actions...))
    return ntuple(i -> flat[offsets[i]+1 : offsets[i+1]], Val(N))
end

function _ipa(
    nums_actions::NTuple{N,Int},
    payoffs::Vector{Float64},
    ray::Vector{Float64},
    zh::Vector{Float64},
    alpha::Float64,
    fuzz::Float64,
) where N
    num_players = Cint(N)
    actions_c = Cint[a for a in nums_actions]
    M = sum(nums_actions)
    ans = zeros(Float64, M)
    ret = ccall(
        (:ipa, libgametracer), Cint,
        (Cint, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble, Ptr{Cdouble}),
        num_players, actions_c, payoffs, ray, zh, alpha, fuzz, ans
    )

    ret < 0 && error("IPA returned shim error code $ret")
    ret == 0 && error("IPA failed: no equilibrium found (ret=0)")  

    return ans
end

function _gnm(
    nums_actions::NTuple{N,Int},
    payoffs::Vector{Float64},
    ray::Vector{Float64},
    steps::Integer,
    fuzz::Float64,
    lnmfreq::Integer,
    lnmmax::Integer,
    lambdamin::Float64,
    wobble::Integer,
    threshold::Float64,
) where N
    num_players = Cint(N)
    actions_c   = Cint[a for a in nums_actions]
    M           = sum(nums_actions)
    answers_ref = Ref{Ptr{Cdouble}}(C_NULL)

    num_eq = ccall(
        (:gnm, libgametracer), Cint,
        (Cint, Ptr{Cint}, Ptr{Cdouble},
         Ptr{Cdouble}, Ref{Ptr{Cdouble}},
         Cint, Cdouble, Cint, Cint, Cdouble, Cint, Cdouble),
        num_players, actions_c, payoffs,
        ray, answers_ref,
        Cint(steps), fuzz, Cint(lnmfreq), Cint(lnmmax),
        lambdamin, Cint(wobble), threshold
    )

    # ret < 0: shim error, answers == NULL
    num_eq < 0 && error("GNM returned shim error code $num_eq")

    # ret == 0: 0 equilibria, answers == NULL
    num_eq == 0 && return Vector{Vector{Float64}}()
    
    # ret > 0: num_eq equilibria, answers is malloc'd buffer
    ptr = answers_ref[]
    ptr != C_NULL || error("GNM returned num_eq=$num_eq but answers pointer was NULL")

    answers = try
        answers_view = unsafe_wrap(Array, ptr, (M, Int(num_eq)); own=false)
        copy(answers_view)
    finally
        _gametracer_free(ptr)
    end

    return [answers[:, j] for j in 1:Int(num_eq)]
end

function _gametracer_free(ptr::Ptr{Cdouble})
    ccall((:gametracer_free, libgametracer), Cvoid, (Ptr{Cvoid},), ptr)
end






end
