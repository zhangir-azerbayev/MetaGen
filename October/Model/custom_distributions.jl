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



# #Object distribution
struct Object_Distribution_Present <: Gen.Distribution{AbstractVector{T}} where {T <: Real} end #ERROR: invalid subtyping in definition of Object_Distribution_Present

const object_distribution_present = Object_Distribution_Present()

function random(::Object_Distribution_Present, mu::AbstractVector{U},
                cov::AbstractMatrix{V}, cat::Int) where {U <: Real, V <: Real}
    [rand(Distributions.MvNormal(mu, cov)), cat]
end

function logpdf(::Object_Distribution_Present, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}, cat::Int) where {T <: Real, U <: Real, V <: Real}

    n = length(x)
    #if category mismatch
    cat!=x[n] && return -Inf
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, x[1:n-1])
end

function logpdf_grad(::Object_Distribution_Present, x::AbstractVector{T}, mu::AbstractVector{U},
    cov::AbstractMatrix{V}, cat::Int) where {T <: Real,U <: Real, V <: Real}
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution_Present)(mu, cov, cat) = Gen.random(Object_Distribution_Present(), mu, cov, cat)

has_output_grad(::Object_Distribution_Present) = false
has_argument_grads(::Object_Distribution_Present) = (false,)

export object_distribution_present
