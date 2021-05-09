const Detection2D = Tuple{Float64, Float64, Int64} #x on image, y on image, category
#returns a line segment of possible 3D positions of the object that oculd have caused this 2D detection
function get_line_segment(camera_params::Camera_Params, params::Video_Params, detection::Detection2D)

    x = detection[1]
    y = detection[2]

    #now (0,0) is the center of the image
    x = x - params.image_dim_x/2
    y = y - params.image_dim_y/2

    x = x / (params.image_dim_x / 2)
    y = y / (params.image_dim_y / 2)

    angle_from_vertical = asin(x * sin(deg2rad(params.horizontal_FoV)))
    angle_from_horizontal = asin(y * sin(deg2rad(params.vertical_FoV))) #in radians

    (a_vertical, b_vertical, c_vertical) = get_vertical_plane(camera_params)
    (a_horizontal, b_horizontal, c_horizontal) = get_horizontal_plane(camera_params, a_vertical, b_vertical, c_vertical)

    #get one more point on the ray. don't care which point, but could be on same plane as camera's focus
    l = get_distance(camera_params.camera_location, camera_params.camera_focus)
    x = l*tan(angle_from_vertical)
    y = l*tan(angle_from_horizontal)
    normalized_v = [a_vertical, b_vertical, c_vertical]./norm([a_vertical, b_vertical, c_vertical])
    normalized_h = [a_horizontal, b_horizontal, c_horizontal]./norm([a_horizontal, b_horizontal, c_horizontal])
    focus = [camera_params.camera_focus.x, camera_params.camera_focus.y, camera_params.camera_focus.z]
    point = focus + x .* normalized_v + y .* normalized_h
    println("point ", point)
    point = Coordinate(point[1], point[2], point[3])

    #check each wall for an intersection within limits
    (endpoint_x, endpoint_y, endpoint_z, a, b, c) = check_walls(point, camera_params.camera_location, params)

    return Line_Segment(camera_params.camera_location, Coordinate(endpoint_x, endpoint_y, endpoint_z), a, b, c)
end

function get_distance(p1::Coordinate, p2::Coordinate)
    return sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2 + (p1.z-p2.z)^2)
end

function check_walls(point::Coordinate, camera::Coordinate, params::Video_Params)
    a = point.x - camera.x
    b = point.y - camera.y
    c = point.z - camera.z

    #get intersection between line and plane

    #check x = x_min
    x = params.x_min
    t = (x - camera.x)/a
    y = camera.y + b*t
    z = camera.z + c*t
    if y < params.y_max && y > params.y_min && z < params.z_max && z > params.z_min && t > 0 #t>0 checks right side
        return (x, y, z, a, b, c)
    end

    #check x = x_max
    x = params.x_max
    t = (x - camera.x)/a
    y = camera.y + b*t
    z = camera.z + c*t
    if y < params.y_max && y > params.y_min && z < params.z_max && z > params.z_min && t > 0
        return (x, y, z, a, b, c)
    end

    #check y = y_min
    y = params.y_min
    t = (y - camera.y)/b
    x = camera.x + a*t
    z = camera.z + c*t
    if x < params.x_max && x > params.x_min && z < params.z_max && z > params.z_min && t > 0
        return (x, y, z, a, b, c)
    end

    #check y = y_max
    y = params.y_max
    t = (y - camera.y)/b
    x = camera.x + a*t
    z = camera.z + c*t
    if x < params.x_max && x > params.x_min && z < params.z_max && z > params.z_min && t > 0
        return (x, y, z, a, b, c)
    end

    #check z = z_min
    z = params.z_min
    t = (z - camera.z)/c
    x = camera.x + a*t
    y = camera.y + b*t
    if x < params.x_max && x > params.x_min && y < params.y_max && y > params.y_min && t > 0
        return (x, y, z, a, b, c)
    end

    #check z = z_max
    z = params.z_max
    t = (z - camera.z)/c
    x = camera.x + a*t
    y = camera.y + b*t
    if x < params.x_max && x > params.x_min && y < params.y_max && y > params.y_min && t > 0
        return (x, y, z, a, b, c)
    end
end

export get_line_segment
