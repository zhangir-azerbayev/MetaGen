using LinearAlgebra

###################################################

"""returns the object's position on the 2D image in pixel space. (0,0) is the center of the image"""
function get_image_xy(camera_params::Camera_Params, params::Video_Params, object::Coordinate)
    if on_right_side(camera_params, object) > 0
        x, y = locate(camera_params, params, object)
    else #on wrong side of camera
        x = Inf
        y = Inf
    end
    return x, y
end

"""
check if on the right side of camera. compare to plane going through camera's
location with a normal vector going from camera to focus
calculating a(x-x_0)+b(y-y_0)+c(z-z_0).
if it turns out positive, it's on the focus' side of the camera
"""
function on_right_side(camera_params::Camera_Params, object::Coordinate)
    c = camera_params.camera_location
    f = camera_params.camera_focus
    (f.x - c.x)*(object.x - c.x) + (f.y - c.y)*(object.y - c.y) + (f.z - c.z)*(object.z - c.z)
end

"""given camera info and object's location, find object's location on 2-D image"""
function locate(camera_params::Camera_Params,
    params::Video_Params, object::Coordinate)

    (a_vertical, b_vertical, c_vertical) = get_vertical_plane(camera_params)
    (a_horizontal, b_horizontal, c_horizontal) = get_horizontal_plane(camera_params, a_vertical, b_vertical, c_vertical)

    #verified working to here.
    #println("here")
    #println("a,b,c vertical ", (a_vertical, b_vertical, c_vertical))

    s_x = object.x-camera_params.camera_location.x
    s_y = object.y-camera_params.camera_location.y
    s_z = object.z-camera_params.camera_location.z



    # object = [object.x, object.y, object.z]
    # camera_focus = [camera_params.camera_focus.x, camera_params.camera_focus.y, camera_params.camera_focus.z]
    # camera_loc = [camera_params.camera_location.x, camera_params.camera_location.y, camera_params.camera_location.z]
    #
    # println("dot ", dot((camera_focus - camera_loc), (object - camera_focus)))

    (s_x_v, s_y_v, s_z_v) = proj_vec_to_plane(a_vertical, b_vertical, c_vertical, s_x, s_y, s_z)
    (s_x_h, s_y_h, s_z_h) = proj_vec_to_plane(a_horizontal, b_horizontal, c_horizontal, s_x, s_y, s_z)

    #println("s_x_v, s_y_v, s_z_v horizontal ", (s_x_v, s_y_v, s_z_v))
    #println("s_x_h, s_y_h, s_z_h horizontal ", (s_x_h, s_y_h, s_z_h))

    angle_from_vertical = get_angle(a_vertical, b_vertical, c_vertical, s_x_h, s_y_h, s_z_h)
    angle_from_horizontal = get_angle(a_horizontal, b_horizontal, c_horizontal, s_x_v, s_y_v, s_z_v)

    #angle_from_vertical = get_angle(a_vertical, b_vertical, c_vertical, s_x, s_y, s_z)
    #angle_from_horizontal = get_angle(a_horizontal, b_horizontal, c_horizontal, s_x, s_y, s_z)

    #println("angle_from_vertical ", rad2deg(angle_from_vertical))
    #println("angle_from_horizontal ", rad2deg(angle_from_horizontal))

    #sin won't differentiate between angles above and below 90 degrees. So 60 degrees and 120 will look the same, but should be okay since can
    #only be between +90 and -90 since on right side of the camera.

    x = params.image_dim_x/2 * (angle_from_vertical / deg2rad(params.horizontal_FoV/2))
    y = params.image_dim_y/2 * (-angle_from_horizontal / deg2rad(params.vertical_FoV/2))

    x = x + params.image_dim_x/2
    y = y + params.image_dim_y/2

    return (x, y)
end

"""
returns the angle between the vector and the plane
#a,b,c is coefficients of plane. x, y, z is the vector.
#whether the sin is positive or negative depends on the normal.
"""
function get_angle(a, b, c, x, y, z)
    numerator = a*x + b*y + c*z #took out abs
    denominator = sqrt(a^2+b^2+c^2) * sqrt(x^2+y^2+z^2)
    return asin(numerator/denominator) #angle in radians
end

"""
returns a,b,c for the vertical plane. only need the camera parameters
y-axis is the upward one
"""
function get_vertical_plane(camera_params::Camera_Params)
    p1 = camera_params.camera_location
    p2 = camera_params.camera_focus
    p3 = Coordinate(p1.x, p1.y+1, p1.z)
    (a, b, c) = get_abc_plane(p1, p2, p3)
    #want normal to be be on the "righthand" side of camera
    #x and z component of vec for the direction camera is pointing
    camera_pointing_x = camera_params.camera_focus.x-camera_params.camera_location.x
    camera_pointing_y = camera_params.camera_focus.y-camera_params.camera_location.y
    camera_pointing_z = camera_params.camera_focus.z-camera_params.camera_location.z

    result = cross([a, b, c], [camera_pointing_x, camera_pointing_y, camera_pointing_z])
    #if y-component < 0, multiply by -1
    to_return = result[2] < 0 ? (-a, -b, -c) : (a, b, c)
    return(to_return)
end

"""need camera parameters and vertical plane (so horizontal will be perpendicular to it)"""
function get_horizontal_plane(camera_params::Camera_Params, a_vertical, b_vertical, c_vertical)
    p1 = camera_params.camera_location
    p2 = camera_params.camera_focus
    p3 = Coordinate(p1.x+a_vertical, p1.y+b_vertical, p1.z+c_vertical) #adding normal vector (a, b, c) to the point to make a third point on the horizontal plane
    (a, b, c) = get_abc_plane(p1, p2, p3)
    #make sure that this normal is upright, so b > 0
    if b < 0
        return(-a, -b, -c)
    elseif b > 0
        return(a, b, c)
    else #when b==0, camera is looking straight up or straight down.
        println("uh oh. camera looking straight up or straight down")
        return(a, b, c)
    end
end

"""
given 3 points on a plane (p1, p2, p3), get a, b, and c coefficients in the general form.
abc also gives a normal vector
"""
function get_abc_plane(p1::Coordinate, p2::Coordinate, p3::Coordinate)
    #using Method 2 from wikipedia: https://en.wikipedia.org/wiki/Plane_(geometry)#:~:text=In%20mathematics%2C%20a%20plane%20is,)%20and%20three%2Ddimensional%20space.
    D = det([p1.x p2.x p3.x; p1.y p2.y p3.y; p1.z p2.z p3.z])

    if D==0
        println("crap! determinant D=0")
        println(p1, p2, p3)
    end
    #implicitly going to say d=-1 to obtain solution set
    a = det([1 1 1; p1.y p2.y p3.y; p1.z p2.z p3.z])/D
    b = det([p1.x p2.x p3.x; 1 1 1; p1.z p2.z p3.z])/D
    c = det([p1.x p2.x p3.x; p1.y p2.y p3.y; 1 1 1])/D

    #normalize / make unit normal vectors
    #a = a / sqrt(a^2 + b^2 + c^2)
    #b = b / sqrt(a^2 + b^2 + c^2)
    #c = c / sqrt(a^2 + b^2 + c^2)
    return (a, b, c)
end

function proj_vec_to_plane(a, b, c, x, y , z)
    num = a * x + b * y + z * c
    denom = a^2 + b^2 + c^2
    constant = num/denom
    u_1 = x - constant * a
    u_2 = y - constant * b
    u_3 = z - constant * c
    return (u_1, u_2, u_3)
end

export get_image_xy
