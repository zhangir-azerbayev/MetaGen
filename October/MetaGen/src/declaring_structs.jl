Base.@kwdef struct Coordinate
    x::Float64
    y::Float64
    z::Float64
end

Base.@kwdef struct Video_Params
    lambda_objects::Float64 = 1
    possible_objects::Vector{Int64} = [1, 2, 3, 4, 5]
    v::Matrix{Float64} = zeros(5, 2)
    x_max::Float64 = 100
    y_max::Float64 = 100
    z_max::Float64 = 100
    num_receptive_fields::Int64 = 1
end

#The camera params that change in a video
Base.@kwdef struct Camera_Params
    camera_location::Coordinate
    camera_focus::Coordinate
end

#Unchanging camera param
Base.@kwdef struct Permanent_Camera_Params
    image_dim_x::Int64 = 320
    image_dim_y::Int64 = 240

    #horizontal field of view
    horizontal_FoV::Float64 = 60
    vertical_FoV::Float64 = 40
end

Base.@kwdef struct Receptive_Field
    p1::Tuple{Int64, Int64} #upper left
    p2::Tuple{Int64, Int64} #lower right
end
