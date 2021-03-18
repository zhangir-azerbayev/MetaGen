Detection2D = Tuple{Float64, Float64, Int64} #x on image, y on image, category

Detection3D = Tuple{Float64, Float64, Float64, Int64} #x, y, z, category

##############################################################################################

struct Multinomial <: Gen.Distribution{Vector{Int}} end

const multinomial = Multinomial()

function Gen.random(::Multinomial, n::Int, probs::AbstractArray{U,1})  where {U <: Real}
    collect(Int64, map(i -> rand(Distributions.Categorical(probs)), 1:n))
end

#n is the number of times sampled
function Gen.logpdf(::Multinomial, x::AbstractArray{Int}, n::Int, probs::AbstractArray{U,1}) where {U <: Real}
    length(x) != n && return -Inf
    sum(map(i -> Gen.logpdf(categorical, i, probs), x))
end

function Gen.logpdf_grad(::Multinomial, x::AbstractArray{Int}, n::Int, probs::AbstractArray{U,1})  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

(::Multinomial)(n, probs) = Gen.random(Multinomial(), n, probs)
is_discrete(::Multinomial) = true

has_output_grad(::Multinomial) = false
has_argument_grads(::Multinomial) = (false,)

export multinomial

##############################################################################################

# #Object distribution for 3D detection
struct Object_Distribution_Present <: Gen.Distribution{Detection3D} end

const object_distribution_present = Object_Distribution_Present()

function Gen.random(::Object_Distribution_Present, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}

    #Detection(rand(Distributions.MvNormal(mu, cov))..., cat)
    rand(Distributions.MvNormal(mu, cov))..., cat
end

function Gen.logpdf(::Object_Distribution_Present, x::Detection3D, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}
    n = length(x)
    #if category mismatch
    cat!=x[n] && return -Inf
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, collect(x[1:n-1]))
end

function Gen.logpdf_grad(::Object_Distribution_Present, x::AbstractVector{V}, mu::AbstractVector{V},
    cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution_Present)(mu, cov, cat) = Gen.random(Object_Distribution_Present(), mu, cov, cat)

has_output_grad(::Object_Distribution_Present) = false
has_argument_grads(::Object_Distribution_Present) = (false,)

export object_distribution_present

##############################################################################################

# #Object distribution for 2D detection
struct Object_Distribution_Image <: Gen.Distribution{Detection2D} end

const object_distribution_image = Object_Distribution_Image()

function Gen.random(::Object_Distribution_Image, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}

    #Detection(rand(Distributions.MvNormal(mu, cov))..., cat)
    rand(Distributions.MvNormal(mu, cov))..., cat
end

function Gen.logpdf(::Object_Distribution_Image, x::Detection2D, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}
    n = length(x)
    #if category mismatch
    cat!=x[n] && return -Inf
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, collect(x[1:n-1]))
end

function Gen.logpdf_grad(::Object_Distribution_Image, x::AbstractVector{V}, mu::AbstractVector{V},
    cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution_Image)(mu, cov, cat) = Gen.random(Object_Distribution_Image(), mu, cov, cat)

has_output_grad(::Object_Distribution_Image) = false
has_argument_grads(::Object_Distribution_Image) = (false,)

export object_distribution_image

##############################################################################################

#Distributions
#TruncatedPoisson
export trunc_poisson

struct TruncatedPoisson <: Gen.Distribution{Int} end

const trunc_poisson = TruncatedPoisson()

function Gen.logpdf(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U) where {U <: Real}
	d = Distributions.Poisson(lambda)
	td = Distributions.Truncated(d, low, high)
	Distributions.logpdf(td, x)
end

function Gen.logpdf_grad(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U)  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::TruncatedPoisson, lambda::U, low::U, high::U)  where {U <: Real}
	d = Distributions.Poisson(lambda)
	rand(Distributions.Truncated(d, low, high))
end

(::TruncatedPoisson)(lambda, low, high) = random(TruncatedPoisson(), lambda, low, high)
is_discrete(::TruncatedPoisson) = true

has_output_grad(::TruncatedPoisson) = false
has_argument_grads(::TruncatedPoisson) = (false,)

export trunc_poisson
