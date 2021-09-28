"A location in 3D space"
Base.@kwdef struct Coordinate
    x::Float64
    y::Float64
    z::Float64
end

"Parametrizes a scene"
Base.@kwdef struct Video_Params
    p_objects::Float64 = 0.9
    n_possible_objects = 8
    possible_objects::Vector{Int64} = collect(1:n_possible_objects)
    probs_possible_objects = collect(ones(n_possible_objects)./n_possible_objects)
    x_min::Float64 = -13/2
    y_min::Float64 = -2 #empirically the camera sometimes looks down at -1.35
    z_min::Float64 = -13/2
    x_max::Float64 = 13/2
    y_max::Float64 = 3
    z_max::Float64 = 13/2
    delta = 0.5 #sd for gaussian #midpoint of logistic curve for prior over distance between objects
    #k = 10. #slope of logistic curve for prior over distance between objects
    num_receptive_fields::Int64 = 1
    #camera parameters that don't change
    image_dim_x::Int64 = 256
    image_dim_y::Int64 = 256
    horizontal_FoV::Float64 = 55
    vertical_FoV::Float64 = 55
end

"Parametrizes the state of the camera, which changes within a scene"
Base.@kwdef struct Camera_Params
    camera_location::Coordinate
    camera_focus::Coordinate
end

Base.@kwdef struct Receptive_Field
    "upper left"
    p1::Tuple{Int64, Int64}
    "lower right"
    p2::Tuple{Int64, Int64}
end

"""
Parametrizes a line segment in 3d space.

``L = \\mathrm{startpoint} + (a, b, c)t``, where ``t\\in [0, 1]``.
``\\mathrm{endpoint} = \\mathrm{startpoint} + (a, b, c)``.
"""
Base.@kwdef struct Line_Segment
    start::Coordinate
    endpoint::Coordinate
    a::Float64
    b::Float64
    c::Float64
end

"""
Stores a probability distribution over the object categories.

Used for defining the proposal function.
"""
Base.@kwdef struct Perturb_Params
    probs_possible_objects::Vector{Float64}
end

export Coordinate
export Video_Params
export Camera_Params
export Permanent_Camera_Params
export Receptive_Field
export Line_Segement
export Vector
export Perturb_Params
