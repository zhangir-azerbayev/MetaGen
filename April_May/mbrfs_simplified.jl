# MULTI-BERNOULLI RFS
# This is a simplified version of Eivinas and Mario' mbrfs.jl
# Don't think I need whatever get_A was about
# Doesn't require Combinatorics package
include("permutations.jl")
include("distribution_params.jl")
include("brfs.jl")

using Gen

export mbrfs, MBRFSParams

abstract type RFS <: Gen.Distribution{Vector} end

struct MBRFS <: RFS end

#DISTRUBITION_PARAMS
abstract type RFSParams end

struct MBRFSParams <: RFSParams
    rs::Vector{Float64}
    rvs::Vector{Gen.Distribution}
    rvs_args::Vector
end

const mbrfs = MBRFS()

function Gen.random(::MBRFS,
                    params::MBRFSParams)
    sample = []

    for i=1:length(params.rs)
        b = brfs(params.rs[i], params.rvs[i], params.rvs_args[i])
        if b != []
            push!(sample, b[1])
        end
    end

	return sample
end

function Gen.logpdf(::MBRFS,
                    xs::Vector,
                    params::MBRFSParams)

    # MBRFS can only support sets <= number of components
    nx = length(xs)
    if nx > length(params.rvs)
        # println("nx > length(params.rvs)")
        return log(0)
    end

    # params = {a, b}
    perms = permutations(1:length(params.rvs)) #how many rvs there are
    # println("perms ", perms)
    lpdfs = fill(-Inf, length(perms))
    for (i, perm) in enumerate(perms)
        lpdf_perm = 0
        # [a, b] -> [?(x1), ?(x2)]
        for (j, element) in enumerate(perm)
            # println("j ", j)
            # println("element ", element)
            # a -> x1
            x = (j <= nx) ? xs[j] : []
            # println("x ", x)
            # println("params.rs[element] ", params.rs[element])
            # println("params.rvs[element] ", params.rvs[element])
            # println("params.rvs_args[element] ", params.rvs_args[element])
            lpdf_perm = lpdf_perm + Gen.logpdf(brfs, x, params.rs[element], params.rvs[element], params.rvs_args[element])
            # println("lpdf_perm ", lpdf_perm)
        end
        # println("lpdf_perm ", lpdf_perm)
        lpdfs[i] = lpdf_perm
    end
    # println("lpdfs ", lpdfs)
    lpdf = logsumexp(lpdfs)
    # println("lpdf ", lpdf)

    return lpdf
end

(::MBRFS)(params) = Gen.random(MBRFS(), params)

Gen.has_output_grad(::MBRFS) = false
Gen.logpdf_grad(::MBRFS, value::Vector, args...) = (nothing,)


# #helper function
# function permutations()
