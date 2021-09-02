"""
Represents an object detection as ``(x_{\\mathrm{midpoint}},
y_{\\mathrm{midpoint}}, \\mathrm{category}``
"""
const Detection2D = Tuple{Float64, Float64, Int64}

"""
Represents an object in the scene as ``(\\mathrm{location}, category)``
"""
const Object3D = Tuple{Float64, Float64, Float64, Int64} #x, y, z, category

export Detection2D
export Object3D

##############################################################################################
#
# struct Multinomial <: Gen.Distribution{Vector{Int}} end
#
# const multinomial = Multinomial()
#
# function Gen.random(::Multinomial, n::Int, probs::AbstractArray{U,1})  where {U <: Real}
#     collect(Int64, map(i -> rand(Distributions.Categorical(probs)), 1:n))
# end
#
# #n is the number of times sampled
# function Gen.logpdf(::Multinomial, x::AbstractArray{Int}, n::Int, probs::AbstractArray{U,1}) where {U <: Real}
#     length(x) != n && return -Inf
#     sum(map(i -> Gen.logpdf(categorical, i, probs), x))
# end
#
# function Gen.logpdf_grad(::Multinomial, x::AbstractArray{Int}, n::Int, probs::AbstractArray{U,1})  where {U <: Real}
# 	gerror("Not implemented")
# 	(nothing, nothing)
# end
#
# (::Multinomial)(n, probs) = Gen.random(Multinomial(), n, probs)
# is_discrete(::Multinomial) = true
#
# has_output_grad(::Multinomial) = false
# has_argument_grads(::Multinomial) = (false,)
#
# export multinomial

##############################################################################################

"Object distribution for 3D detection"
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

"Object distribution for 2D detection"
struct Object_Distribution_Image <: Gen.Distribution{Detection2D} end

const object_distribution_image = Object_Distribution_Image()

function Gen.random(::Object_Distribution_Image, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}

    #Detection(rand(Distributions.MvNormal(mu, cov))..., cat)
    rand(Distributions.MvNormal(mu, cov))..., cat
end

function Gen.logpdf(::Object_Distribution_Image, x::Detection2D, mu::AbstractVector{V},
                cov::AbstractMatrix{V}, cat::Int64) where {V <: Float64}
    #if category mismatch
    cat!=x[3] && return -Inf
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, collect(x[1:2]))
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
"For a new 3-D object placed anywhere"
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

"Little helper function for constructing 3D objects"
function construct_3D(cat::Int64, params::Video_Params)
    x = uniform(params.x_min, params.x_max)
    y = uniform(params.y_min, params.y_max)
    z = uniform(params.z_min, params.z_max)
    return (x, y, z, cat)
end

##############################################################################################
"For populating the scene with objects spaced out from one another with a graded, Gaussian prior"
struct Object_Distribution_Gaussian <: Gen.Distribution{Array{Object3D}} end

const object_distribution_gaussian = Object_Distribution_Gaussian()

function Gen.random(::Object_Distribution_Gaussian, params::Video_Params)
    num_objects = geometric(params.p_objects)
    placed_objects = Array{Object3D}(undef, 0)
    for i = 1:num_objects
        cat = categorical(params.probs_possible_objects)
        candidate_obj = construct_3D(cat, params) #placed uniformly randomly
        j = 1 #indexes through the objects that have already been placed and accepted
        while j <= length(placed_objects)
            #resample with probabilty given by Gaussian with distance
            d = dist(candidate_obj, placed_objects[j])
            p = exp(-0.5*(d/params.delta)^2) #one when dist==0
            if bernoulli(p) #if too close, resample
                candidate_obj = construct_3D(cat, params) #placed uniformly randomly
                j = 1 #restart loop over objects already placed
            else
                j = j + 1
            end
        end
        push!(placed_objects, candidate_obj)
    end
    return placed_objects
end

function Gen.logpdf(::Object_Distribution_Gaussian, placed_objects::Array{Object3D}, params::Video_Params)
    num_objects = length(placed_objects)
    p_geometric = Gen.logpdf(geometric, num_objects, params.p_objects)

    if num_objects == 0
        return p_geometric
    end

    #iterate over all pairwise combindations of objects and get distances
    p_distance = 0
    for c in combinations(placed_objects, 2)
        d = dist(c[1], c[2])
        p = exp(-0.5*(d/params.delta)^2) #probability of resampling
        inv = 1-p #probability of sticking with it
        log_inv = log(inv)
        p_distance = p_distance + log_inv
    end

    #categorical
    cats = last.(placed_objects) #grabbing the category types
    p_cats = sum(map(i -> Gen.logpdf(categorical, i, params.probs_possible_objects), cats))

    xs = first.(placed_objects)
    p_xs = sum(map(x -> Gen.logpdf(uniform, x, params.x_min, params.x_max), xs))

    ys = (a->a[2]).(placed_objects)
    p_ys = sum(map(y -> Gen.logpdf(uniform, y, params.y_min, params.y_max), ys))

    zs = (a->a[3]).(placed_objects)
    p_zs = sum(map(z -> Gen.logpdf(uniform, z, params.z_min, params.z_max), zs))

    return p_geometric + p_cats + p_xs + p_ys + p_zs + p_distance
end

function Gen.logpdf_grad(::Object_Distribution_Gaussian, objects_3D::Object3D, params::Video_Params)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution_Gaussian)(params) = Gen.random(Object_Distribution_Gaussian(), params)

has_output_grad(::Object_Distribution_Gaussian) = false
has_argument_grads(::Object_Distribution_Gaussian) = (false,)

export object_distribution_gaussian
##############################################################################################
"For populating the scene with objects spaced out from one another. no two object can be within distance delta of each other"
struct Object_Distribution_Delta <: Gen.Distribution{Array{Object3D}} end

const object_distribution_delta = Object_Distribution_Delta()

function Gen.random(::Object_Distribution_Delta, params::Video_Params)
    num_objects = geometric(params.p_objects)
    placed_objects = Array{Object3D}(undef, 0)
    for i = 1:num_objects
        cat = categorical(params.probs_possible_objects)
        candidate_obj = construct_3D(cat, params) #placed uniformly randomly
        j = 1 #indexes through the objects that have already been placed and accepted
        while j <= length(placed_objects)
            if dist(candidate_obj, placed_objects[j]) < params.delta #if too close, resample
                candidate_obj = construct_3D(cat, params) #placed uniformly randomly
                j = 1 #restart loop over objects already placed
            else
                j = j + 1
            end
        end
        push!(placed_objects, candidate_obj)
    end
    return placed_objects
end

function Gen.logpdf(::Object_Distribution_Delta, placed_objects::Array{Object3D}, params::Video_Params)
    #either the objects are spaced out enough or they aren't.
    #iterate over all pairwise combindations of objects
    for c in combinations(placed_objects, 2)
        if dist(c[1], c[2]) < params.delta
            #println("c ", c)
            #println("c[1] ", c[1])
            #println("c[2] ", c[2])
            return -Inf #if the objects aren't far enough apart, log probability is -Inf
        end
    end

    num_objects = length(placed_objects)
    p_geometric = Gen.logpdf(geometric, num_objects, params.p_objects)

    if num_objects == 0
        return p_geometric
    end

    #categorical
    cats = last.(placed_objects) #grabbing the category types
    p_cats = sum(map(i -> Gen.logpdf(categorical, i, params.probs_possible_objects), cats))

    xs = first.(placed_objects)
    p_xs = sum(map(x -> Gen.logpdf(uniform, x, params.x_min, params.x_max), xs))

    ys = (a->a[2]).(placed_objects)
    p_ys = sum(map(y -> Gen.logpdf(uniform, y, params.y_min, params.y_max), ys))

    zs = (a->a[3]).(placed_objects)
    p_zs = sum(map(z -> Gen.logpdf(uniform, z, params.z_min, params.z_max), zs))

    return p_geometric + p_cats + p_xs + p_ys + p_zs
end

function Gen.logpdf_grad(::Object_Distribution_Delta, objects_3D::Object3D, params::Video_Params)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution_Delta)(params) = Gen.random(Object_Distribution_Delta(), params)

has_output_grad(::Object_Distribution_Delta) = false
has_argument_grads(::Object_Distribution_Delta) = (false,)

export object_distribution_delta

#tiny helper function for getting distance between two objects
function dist(a::Object3D, b::Object3D)
    sqrt((a[1]-b[1])^2 + (a[2]-b[2])^2 + (a[3]-b[3])^2)
end

##############################################################################################
"For a new 3-D object placed anywhere. category already determined"
struct Location_Distribution_Uniform <: Gen.Distribution{Object3D} end

const location_distribution_uniform = Location_Distribution_Uniform()

function Gen.random(::Location_Distribution_Uniform, cat::Int64, params::Video_Params)
    objects_3D = construct_3D(cat, params)
end

function Gen.logpdf(::Location_Distribution_Uniform, object_3D::Object3D, cat::Int64, params::Video_Params)
    #could add something making sure category is right.

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

    p_x + p_y + p_z
end

function Gen.logpdf_grad(::Location_Distribution_Uniform, objects_3D::Object3D, cat::Int64, params::Video_Params)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Location_Distribution_Uniform)(cat, params) = Gen.random(Location_Distribution_Uniform(), cat, params)

has_output_grad(::Location_Distribution_Uniform) = false
has_argument_grads(::Location_Distribution_Uniform) = (false,)

export location_distribution_uniform

##############################################################################################
"For a new 3-D object placed along certain line segments with Gaussian noise, category has already been chosen"
struct New_Location_Distribution_Noisy <: Gen.Distribution{Object3D} end

const new_location_distribution_noisy = New_Location_Distribution_Noisy()

function Gen.random(::New_Location_Distribution_Noisy, cat::Int64, params::Video_Params, line_segments::Array{Line_Segment})
    n = size(line_segments)[1]
    if n < 0
        println("problem. sampling location from new_location_distribution_noisy when we shouldn't be")
    end
    i = categorical(fill(1/n, n)) #which line segment
    line_segment = line_segments[i]
    #println("line_segment ", line_segment)
    #length of the line segment
    length = sqrt((line_segment.start.x-line_segment.endpoint.x)^2 + (line_segment.start.y-line_segment.endpoint.y)^2 + (line_segment.start.z-line_segment.endpoint.z)^2)
    #println("length ", length)
    d = uniform(0, length) #sample a distance along the line segment
    #println("d ", d)
    #plug in d/length as t
    x = line_segment.start.x + line_segment.a*(d/length)
    y = line_segment.start.y + line_segment.b*(d/length)
    z = line_segment.start.z + line_segment.c*(d/length)

    #add noise
    dist = trunc_normal(0., 0.1, 0., 10.) #mean noise is 0, sd is 0.1?
    #println("dist ", dist)
    angle = uniform(0, 2*pi)

    (x_noisy, y_noisy, z_noisy) = sample_point(line_segment, Coordinate(x,y,z), dist, angle)

    return (x_noisy, y_noisy, z_noisy, cat)
end

function Gen.logpdf(::New_Location_Distribution_Noisy, object_3D::Object3D, cat::Int64, params::Video_Params, line_segments::Array{Line_Segment})
    #check how many line segments have this point on it
    n = length(line_segments)
    point = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    ps = map(prob_given_line_segment, fill(point, n), line_segments)
    p_point = log(1/n) + log(sum(ps)) #could do something with categorical distibution, but don't know how
    return p_point
end

function Gen.logpdf_grad(::New_Location_Distribution_Noisy, objects_3D::Object3D, cat::Int64, params::Video_Params, line_segments::Array{Line_Segment})
    gerror("Not implemented")
    (nothing, nothing)
end

(::New_Location_Distribution_Noisy)(cat, params, line_segments) = Gen.random(New_Location_Distribution_Noisy(), cat, params, line_segments)

has_output_grad(::New_Location_Distribution_Noisy) = false
has_argument_grads(::New_Location_Distribution_Noisy) = (false,)

function sample_point(line_segment::Line_Segment, point_on_line::Coordinate, dist::Float64, angle::Float64)

    #first, sample a vector on the plane. the plane is defined by being normal to the line segment and including the point
    #point_on_line. describe this plane in point normal form. pick any two values of x and y and then solve for z to get
    #another point on this plane. then a vector on this plane is given by (x-point_on_line_x, y-point_on_line_y, z-point_on_line_z)
    x = 1
    y = 1
    z = (-line_segment.a*(x-point_on_line.x) - line_segment.b*(y-point_on_line.x))/line_segment.c + point_on_line.y
    v1 = [x-point_on_line.x, y-point_on_line.y, z-point_on_line.z]

    #get v2, which is perpendicular to the line segement and to v1, bu taking the cross product.
    v2 = cross(v1, [line_segment.a, line_segment.b, line_segment.c])

    #make unit vectors
    v1 = v1./sqrt(sum(v1.^2))
    v2 = v2./sqrt(sum(v2.^2))

    point = [point_on_line.x, point_on_line.y, point_on_line.z] + dist * sin(angle) .*v1 + dist * cos(angle) .*v2
    return (point[1], point[2], point[3])
end

#gives the probability of a point given a segment
function prob_given_line_segment(point::Coordinate, line_segment::Line_Segment)

    #point is on a plane normal to line segment. that's given by point normal form. finding intersection between
    #this plane and the line segment by solving for t
    t = (line_segment.a*(point.x - line_segment.start.x) + line_segment.b*(point.y - line_segment.start.y) + line_segment.c*(point.z - line_segment.start.z))/(line_segment.a^2 + line_segment.b^2 + line_segment.c^2)
    if t < 0 || t > 1 #would mean that this is off the line segment
        return 0
    end

    line_seg_coef = [line_segment.a, line_segment.b, line_segment.c]
    line_seg_start = [line_segment.start.x, line_segment.start.y, line_segment.start.z]
    point_on_line = t.*line_seg_coef .+ line_seg_start

    #get distance between point_on_line and point
    point = [point.x, point.y, point.z]
    dist = sqrt(sum((point - point_on_line).^2)) #norm might do the same

    #actually calculate probabilities
    len = sqrt((line_segment.start.x-line_segment.endpoint.x)^2 + (line_segment.start.y-line_segment.endpoint.y)^2 + (line_segment.start.z-line_segment.endpoint.z)^2)
    p1 = Gen.logpdf(uniform, 0, 0, len)#doesn't matter what the actual value is because it's uniform, so plug in 0
    p2 = Gen.logpdf(trunc_normal, dist, 0., 0.1, 0., 10.)#make sure it matches above
    p3 = Gen.logpdf(uniform, 0, 0, 2*pi)

    return MathConstants.e^(p1 + p2 + p3) #transform it back to probability space
end

export new_location_distribution_noisy

##############################################################################################
"For changing the location of an object: either by Gaussian noise based on previous location, or resampled along certain line segments with Gaussian noise, category has already been chosen"
struct New_Location_Distribution_Noisy_Or_Gaussian <: Gen.Distribution{Object3D} end

const new_location_distribution_noisy_or_gaussian = New_Location_Distribution_Noisy_Or_Gaussian()

function Gen.random(::New_Location_Distribution_Noisy_Or_Gaussian, mu::Vector{Float64},
    cov::Matrix{Float64}, cat::Int64, params::Video_Params, line_segments::Array{Array{Line_Segment,1},1})

    if length(line_segments[cat]) > 0
        #coin flip.
        if bernoulli(0.5)
            #println("from data-driven distribution")
            to_return = new_location_distribution_noisy(cat, params, line_segments[cat])
        else
            #println("from gaussian around previous location")
            to_return = object_distribution_present(mu::AbstractVector{Float64}, cov::AbstractMatrix{Float64}, cat::Int64)
        end
    else
        to_return = object_distribution_present(mu::AbstractVector{Float64}, cov::AbstractMatrix{Float64}, cat::Int64)
    end

    return to_return
end

function Gen.logpdf(::New_Location_Distribution_Noisy_Or_Gaussian, x::Object3D, mu::AbstractVector{Float64},
    cov::AbstractMatrix{Float64}, cat::Int64, params::Video_Params, line_segments::Array{Array{Line_Segment,1},1})

    p_from_gaussian = Gen.logpdf(object_distribution_present, x, mu, cov, cat)

    if length(line_segments) < 0
        return p_from_gaussian
    else
        p_from_noisy = Gen.logpdf(new_location_distribution_noisy, x, cat, params, line_segments[cat])
        return log(0.5*(MathConstants.e^p_from_noisy + MathConstants.e^p_from_gaussian))
    end
end

function Gen.logpdf_grad(::New_Location_Distribution_Noisy_Or_Gaussian, mu::AbstractVector{Float64},
    cov::AbstractMatrix{Float64}, cat::Int64, params::Video_Params, line_segments::Array{Line_Segment})
    gerror("Not implemented")
    (nothing, nothing)
end

(::New_Location_Distribution_Noisy_Or_Gaussian)(mu, cov, cat, params, line_segments) = Gen.random(New_Location_Distribution_Noisy_or_Gaussian(), mu, cov, cat, params, line_segments)

has_output_grad(::New_Location_Distribution_Noisy_Or_Gaussian) = false
has_argument_grads(::New_Location_Distribution_Noisy_Or_Gaussian) = (false,)

export new_location_distribution_noisy_or_gaussian


##############################################################################################
#Sample the category. If the object category has never been observed, sample location from a uniform distribution
#If it has been observed, sample location with 10% chance from a uniform or from new_object_distribution_noisy
struct New_Object_Distribution_Noisy_Or_Uniform <: Gen.Distribution{Object3D} end

const new_object_distribution_noisy_or_uniform = New_Object_Distribution_Noisy_Or_Uniform()

function Gen.random(::New_Object_Distribution_Noisy_Or_Uniform, params::Video_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
    cat = categorical(params.probs_possible_objects)
    line_segments = line_segments_per_category[cat]

    if length(line_segments) > 0
        coin_flip = bernoulli(0.9)
        if coin_flip
            #println("from data-driven distribution")
            to_return = new_location_distribution_noisy(cat, params, line_segments)
        else
            #println("from uniform distribution")
            to_return = location_distribution_uniform(cat, params)
        end
    else
        #println("from uniform distribution")
        to_return = location_distribution_uniform(cat, params)
    end
    return to_return
end

function Gen.logpdf(::New_Object_Distribution_Noisy_Or_Uniform, object_3D::Object3D, params::Video_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
    cat = object_3D[4]
    line_segments = line_segments_per_category[cat]
    p_cat = Gen.logpdf(categorical, cat, params.probs_possible_objects)

    if length(line_segments) > 0
        p_from_noisy = Gen.logpdf(new_location_distribution_noisy, object_3D, cat, params, line_segments)
        p_from_uniform = Gen.logpdf(location_distribution_uniform, object_3D, cat, params)
        return p_cat + log(0.9*(MathConstants.e^p_from_noisy + MathConstants.e^p_from_uniform)) #0.5 based on p in coin_flip bernoulli
    else #no chance from new_location_distribution_noisy
        p_from_uniform = Gen.logpdf(location_distribution_uniform, object_3D, cat, params)
        return p_cat + p_from_uniform
    end
end

function Gen.logpdf_grad(::New_Object_Distribution_Noisy_Or_Uniform, objects_3D::Object3D, params::Video_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
    gerror("Not implemented")
    (nothing, nothing)
end

(::New_Object_Distribution_Noisy_Or_Uniform)(params, line_segments_per_category) = Gen.random(New_Object_Distribution_Noisy_Or_Uniform(), params, line_segments_per_category)

has_output_grad(::New_Object_Distribution_Noisy_Or_Uniform) = false
has_argument_grads(::New_Object_Distribution_Noisy_Or_Uniform) = (false,)

export new_object_distribution_noisy_or_uniform

##############################################################################################
#For a new 3-D object with the same location but new category
struct Object_Distribution_Category <: Gen.Distribution{Object3D} end

const object_distribution_category = Object_Distribution_Category()

function Gen.random(::Object_Distribution_Category, x::Float64, y::Float64, z::Float64, perturb_params::Perturb_Params)

    cat = categorical(perturb_params.probs_possible_objects)

    return (x, y, z, cat)
end

function Gen.logpdf(::Object_Distribution_Category, object_3D::Object3D, x::Float64, y::Float64, z::Float64, perturb_params::Perturb_Params)
    #categorical
    cat = object_3D[4] #grabbing the category type
    Gen.logpdf(categorical, cat, perturb_params.probs_possible_objects)
end

function Gen.logpdf_grad(::Object_Distribution_Category, objects_3D::Object3D, perturb_params::Perturb_Params)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Object_Distribution_Category)(x, y, z, perturb_params) = Gen.random(Object_Distribution_Category(), x, y, z, perturb_params)

has_output_grad(::Object_Distribution_Category) = false
has_argument_grads(::Object_Distribution_Category) = (false,)

export new_object_distribution_category

##############################################################################################
struct Hallucination_Distribution <: Gen.Distribution{Detection2D} end

const hallucination_distribution = Hallucination_Distribution()

function Gen.random(::Hallucination_Distribution, params::Video_Params, v::Matrix{Real}, rec_field::Receptive_Field)

    #saying there must be no fewer than 0 objects per scene, and at most 100
    #numObjects = @trace(trunc_poisson(sum(params.v[:,1]), -1.0, 100.0), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #objects = @trace(multinomial_objects(numObjects,[0.2,0.2,0.2,0.2,0.2], ), (:objects))
    ws = @inbounds 1.0/sum(v[:,1]) * v[:,1]
    c = categorical(ws)
    detection_2D = construct_2D(c, params, rec_field)
end

function Gen.logpdf(::Hallucination_Distribution, detection_2D::Detection2D, params::Video_Params, v::Matrix{Real}, rec_field::Receptive_Field)

    #multinomial
    cat = detection_2D[3] #grabbing the category type
    ws = @inbounds 1.0/sum(v[:,1]) * v[:,1]
    p_categorical = Gen.logpdf(categorical, cat, ws) #prob_vec is the hallucination lambdas normalized


    #x-coordinate
    #all the x_coordinates
    x = detection_2D[1]
    p_x = Gen.logpdf(uniform, x, rec_field.p1[1], rec_field.p2[1])

    #y-coordinate
    y = detection_2D[2]
    p_y = Gen.logpdf(uniform, y, rec_field.p1[2], rec_field.p2[2])

    p_categorical + p_x + p_y
end

function Gen.logpdf_grad(::Hallucination_Distribution, detection_2D::Detection2D, params::Video_Params, v::Matrix{Real}, rec_field::Receptive_Field)
    gerror("Not implemented")
    (nothing, nothing)
end

(::Hallucination_Distribution)(params, v, rec_field) = Gen.random(Hallucination_Distribution(), params, v, rec_field)

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

function Gen.random(::TruncatedNormal, mu::U, std::U, low::U, high::U)  where {U <: Real}
	n = Distributions.Normal(mu, std)
	rand(Distributions.Truncated(n, low, high))
end

function Gen.logpdf(::TruncatedNormal, x::U, mu::U, std::U, low::U, high::U) where {U <: Real}
	n = Distributions.Normal(mu, std)
	tn = Distributions.Truncated(n, low, high)
	Distributions.logpdf(tn, x)
end

function Gen.logpdf_grad(::TruncatedNormal, x::U, mu::U, std::U, low::U, high::U)  where {U <: Real}
    precision = 1. / (std * std)
    diff = mu - x
    deriv_x = diff * precision
    deriv_mu = -deriv_x
    deriv_std = -1. / std + (diff * diff) / (std * std * std)

    if x<=low
        deriv_x = log(0.001) #trying to have a very small positive gradient
    elseif x>=high
        deriv_x = log(-0.001) #trying to have a very small negative gradient
    end

    (deriv_x, deriv_mu, deriv_std)
end

(::TruncatedNormal)(mu, std, low, high) = random(TruncatedNormal(), mu, std, low, high)
is_discrete(::TruncatedNormal) = false
has_output_grad(::TruncatedNormal) = true
has_argument_grads(::TruncatedNormal) = (true, true, false, false) #just has output gradients for mu and std
