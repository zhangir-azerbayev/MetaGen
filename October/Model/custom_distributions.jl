struct Multinomial <: Gen.Distribution{Vector{Int}} end

const multinomial = Multinomial()

#N is the number of times sampled
function Gen.logpdf(::Multinomial, x::AbstractArray{Int}, N::Int, probs::AbstractArray{U,1}) where {U <: Real}
    if length(x) != N
        return 0
    end
    p = 1
    d = Distributions.Categorical(probs)
    for i in 1:length(x)
        temp = Distributions.logpdf(d, x[i], probs)
        p = p*temp
    end
    return p
end

function Gen.logpdf_grad(::Multinomial, x::AbstractArray{Int}, N::Int, probs::AbstractArray{U,1})  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::Multinomial, N::Int, probs::AbstractArray{U,1})  where {U <: Real}
	x = Vector{Int}(undef, N)
    for i in 1:N
        x[i] = rand(Distributions.Categorical(probs))
    end
    return x
end

(::Multinomial)(N, probs) = random(Multinomial(), N, probs)
is_discrete(::Multinomial) = true

has_output_grad(::Multinomial) = false
has_argument_grads(::Multinomial) = (false,)

export multinomial



# #Object distribution
struct objectdistribution <: Distribution{Vector{T}} where {T <: Real} end

const objectdistribution = ObjectDistribution()

function logpdf(::ObjectDistribution, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}, cat::Int) where {T <: Real, U <: Real, V <: Real}

    n = length(x)
    #if category mismatch
    if cat!=x[n]
        return 0
    end
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, x[1:n-1])
end

function logpdf_grad(::ObjectDistribution, x::AbstractVector{T}, mu::AbstractVector{U},
    cov::AbstractMatrix{V}, cat::Int) where {T <: Real,U <: Real, V <: Real}
    gerror("Not implemented")
    (nothing, nothing)
end

function random(::MultivariateNormal, mu::AbstractVector{U},
                cov::AbstractMatrix{V}, cat::Int) where {U <: Real, V <: Real}
    [rand(Distributions.MvNormal(mu, cov)), cat]
end

(::ObjectDistribution)(mu, cov, cat) = random(ObjectDistribution(), mu, cov, cat)

has_output_grad(::ObjectDistribution) = false
has_argument_grads(::ObjectDistribution) = (false,)

export objectdistribution
