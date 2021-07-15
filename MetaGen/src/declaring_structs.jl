"A location in 3D space"
Base.@kwdef struct Coordinate
    x::Float64
    y::Float64
    z::Float64
end

"Parametrizes a scene"
Base.@kwdef struct Video_Params
    p_objects::Float64 = 0.9
    possible_objects::Vector{Int64} = collect(1:5)
    probs_possible_objects = collect(ones(5)./5)
    x_min::Float64 = -16
    y_min::Float64 = -10
    z_min::Float64 = -10
    x_max::Float64 = 10
    y_max::Float64 = 10
    z_max::Float64 = 10
    num_receptive_fields::Int64 = 1
    #camera parameters that don't change
    image_dim_x::Int64 = 320
    image_dim_y::Int64 = 240
    horizontal_FoV::Float64 = 60
    vertical_FoV::Float64 = 45
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
