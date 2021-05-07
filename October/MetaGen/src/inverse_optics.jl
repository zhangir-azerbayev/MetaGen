const Detection2D = Tuple{Float64, Float64, Int64} #x on image, y on image, category

Base.@kwdef struct line_segment
    x1::Float64
    y1::Float64
    z1::Float64
    x2::Float64
    y2::Float64
    z1::Float64
end

#returns a ray.
function get_image_xy(camera_params::Camera_Params, params::Video_Params, detection::Detection2D)


end
