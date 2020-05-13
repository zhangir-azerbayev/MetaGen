#Mario and Eivinas implemented bernoulli random finite sets

# BERNOULLI RFS

export brfs

struct BRFS <: Gen.Distribution{Vector} end

const brfs = BRFS()

function Gen.random(::BRFS, r::Float64, rv::Gen.Distribution,
					rv_args::Tuple)
    sample = Gen.bernoulli(r) ? [random(rv, rv_args...)] : []
	return sample
end

function Gen.logpdf(::BRFS, x::Vector, r::Float64, rv::Gen.Distribution,
					rv_args::Tuple)

	# println("in brfs logpdf")

    lpdf = 0.0
    if length(x) == 0
		lpdf = log(1-r)
		# println("in if")
	# elseif length(x) == 1
	# 	lpdf = log(r) + Gen.logpdf(rv, first(x), rv_args...)
	elseif length(x) == 2 #modified for when the rv distribution returns two values
		# println("in elseif")
		# println(x)
		# println(r)
		# println(rv)
		# println(rv_args)
		# println("Gen.logpdf(rv, x, rv_args) ", Gen.logpdf(rv, x, rv_args...))
		#lpdf = log(r) + Gen.logpdf(rv, x, rv_args...)
		lpdf = log(r) + Gen.logpdf(rv, x, rv_args...)
	else
		# println("in else")
		lpdf = log(0)
	end

	# println("lpdf ", lpdf)
    #println("brfs lpdf: $lpdf")
    #if isinf(lpdf)
    #    println(x)
    #    println(r)
    #    println(rv)
    #    println(rv_args)
    #end
    return lpdf
end

(::BRFS)(r, rv, rv_args) = Gen.random(BRFS(), r, rv, rv_args)

Gen.has_output_grad(::BRFS) = false
Gen.logpdf_grad(::BRFS, value::Vector, args...) = (nothing,)
