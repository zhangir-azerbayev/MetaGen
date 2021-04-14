const Detection2D = Tuple{Float64, Float64, Int64} #x on image, y on image, category
const Object3D = Tuple{Float64, Float64, Float64, Int64} #x, y, z, category

export Detection2D
export Object3D

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
struct Object_Distribution_Present <: Gen.Distribution{Object3D} end

const object_distribution_present = Object_Distribution_Present()

function Gen.random(::Object_Distribution_Present, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}

    #Detection(rand(Distributions.MvNormal(mu, cov))..., cat)
    rand(Distributions.MvNormal(mu, cov))..., cat
end

function Gen.logpdf(::Object_Distribution_Present, x::Object3D, mu::AbstractVector{V},
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
#For a new 3-D object placed anywhere
struct Object_Distribution <: Gen.Distribution{Object3D} end

const object_distribution = Object_Distribution()

function Gen.random(::Object_Distribution, params::Video_Params)

    c = categorical(params.probs_possible_objects)
    objects_3D = construct_3D(c, params)
end

function Gen.logpdf(::Object_Distribution, object_3D::Object3D, params::Video_Params)
    #categorical
    cat = object_3D[4] #grabbing the category type
    p_categorical = Gen.logpdf(categorical, cat, params.probs_possible_objects)

    #x-coordinate
    #all the x_coordinates
    x = object_3D[1]
    p_x = Gen.logpdf(uniform, x, params.x_min, params.x_max)

    #y-coordinate
    y = object_3D[2]
    p_y = Gen.logpdf(uniform, y, params.y_min, params.y_max)

    #y-coordinate
    z = object_3D[3]
    p_z = Gen.logpdf(uniform, z, params.z_min, params.z_max)

    p_categorical + p_x + p_y + p_z
end

function Gen.logpdf_grad(::Object_Distribution, objects_3D::Object3D, params::Video_Params)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution)(params) = Gen.random(Object_Distribution(), params)

has_output_grad(::Object_Distribution) = false
has_argument_grads(::Object_Distribution) = (false,)

export object_distribution

#little helper function for constructing 3D objects
function construct_3D(cat::Int64, params::Video_Params)
    x = uniform(params.x_min, params.x_max)
    y = uniform(params.y_min, params.y_max)
    z = uniform(params.z_min, params.z_max)
    return (x, y, z, cat)
end

##############################################################################################
struct Hallucination_Distribution <: Gen.Distribution{Detection2D} end

const hallucination_distribution = Hallucination_Distribution()

function Gen.random(::Hallucination_Distribution, params::Video_Params, rec_field::Receptive_Field)

    #saying there must be no fewer than 0 objects per scene, and at most 100
    #numObjects = @trace(trunc_poisson(sum(params.v[:,1]), -1.0, 100.0), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #objects = @trace(multinomial_objects(numObjects,[0.2,0.2,0.2,0.2,0.2], ), (:objects))
    c = categorical(params.v[:,1]./sum(params.v[:,1]))
    detection_2D = construct_2D(c, params, rec_field)
end

function Gen.logpdf(::Hallucination_Distribution, detection_2D::Detection2D, params::Video_Params, rec_field::Receptive_Field)

    #multinomial
    cat = detection_2D[3] #grabbing the category type
    p_categorical = Gen.logpdf(categorical, cat, params.v[:,1]./sum(params.v[:,1])) #prob_vec is the hallucination lambdas normalized

    #x-coordinate
    #all the x_coordinates
    x = detection_2D[1]
    p_x = Gen.logpdf(uniform, x, rec_field.p1[1], rec_field.p2[1])

    #y-coordinate
    y = detection_2D[2]
    p_y = Gen.logpdf(uniform, y, rec_field.p1[2], rec_field.p2[2])

    p_categorical + p_x + p_y
end

function Gen.logpdf_grad(::Hallucination_Distribution, detection_2D::Detection2D, params::Video_Params)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Hallucination_Distribution)(params, rec_field) = Gen.random(Hallucination_Distribution(), params, rec_field)

has_output_grad(::Hallucination_Distribution) = false
has_argument_grads(::Hallucination_Distribution) = (false,)

export hallucination_distribution


#little helper function for constructing 2D objects
function construct_2D(cat::Int64, params::Video_Params, rec_field::Receptive_Field)
    x = uniform(rec_field.p1[1], rec_field.p2[1])
    y = uniform(rec_field.p1[2], rec_field.p2[2])
    return (x, y, cat)
end

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

##############################################################################################
#TruncatedNormal
export trunc_normal

#small issue where all of the inputs need to by Float64. Doesn't accept Int64s
struct TruncatedNormal <: Gen.Distribution{Float64} end

const trunc_normal = TruncatedNormal()

function Gen.logpdf(::TruncatedNormal, x::U, mu::U, std::U, low::U, high::U) where {U <: Real}
	n = Distributions.Normal(mu, std)
	tn = Distributions.Truncated(n, low, high)
	Distributions.logpdf(tn, x)
end

function Gen.logpdf_grad(::TruncatedNormal, x::U, mu::U, std::U, low::U, high::U)  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::TruncatedNormal, mu::U, std::U, low::U, high::U)  where {U <: Real}
	n = Distributions.Normal(mu, std)
	rand(Distributions.Truncated(n, low, high))
end

(::TruncatedNormal)(mu, std, low, high) = random(TruncatedNormal(), mu, std, low, high)
is_discrete(::TruncatedNormal) = false
has_output_grad(::TruncatedPoisson) = false
has_argument_grads(::TruncatedPoisson) = (false,)
