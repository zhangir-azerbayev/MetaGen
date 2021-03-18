#This file contains functions for solving the geometry / optics problem
#of figuring out where the images of objects will be on the camera's 2D photo

using LinearAlgebra

include("declaring_structs.jl")

function get_image_xy(camera_params, permanent_camera_params, object)
    if on_right_side(camera_params, object) > 0
        x, y = locate(camera_params, permanent_camera_params, object)
    else #on wrong side of camera
        #pretend asin is 90. putting it on corner of giant image
        pixels_per_degree_x = permanent_camera_params.image_dim_x/permanent_camera_params.horizontal_FoV
        x = 90 * pixels_per_degree_x
        pixels_per_degree_y = permanent_camera_params.image_dim_y/permanent_camera_params.vertical_FoV
        y = 90 * pixels_per_degree_y
    end
    return x, y
end

#check if on the right side of camera. compare to plane going through camera's
#location with a normal vector going from camera to focus
#calculating a(x-x_0)+b(y-y_0)+c(z-z_0).
#if it turns out positive, it's on the focus' side of the camera
function on_right_side(camera_params::Camera_Params, object::Coordinate)
    c = camera_params.camera_location
    f = camera_params.camera_focus
    (f.x - c.x)*(object.x - c.x) + (f.y - c.y)*(object.y - c.y) + (f.z - c.z)*(object.z - c.z)
end

#given camera info and object's location, find object's location on 2-D image
function locate(camera_params::Camera_Params,
    permanent_camera_params::Permanent_Camera_Params, object::Coordinate)

    (a_vertical, b_vertical, c_vertical) = get_vertical_plane(camera_params)
    (a_horizontal, b_horizontal, c_horizontal) = get_horizontal_plane(camera_params, a_vertical, b_vertical, c_vertical)

    s_x = object.x-camera_params.camera_location.x
    s_y = object.y-camera_params.camera_location.y
    s_z = object.z-camera_params.camera_location.z

    angle_from_vertical = rad2deg(get_angle(a_vertical, b_vertical, c_vertical, s_x, s_y, s_z))
    angle_from_horizontal = rad2deg(get_angle(a_horizontal, b_horizontal, c_horizontal, s_x, s_y, s_z))

    pixels_per_degree_x = permanent_camera_params.image_dim_x/permanent_camera_params.horizontal_FoV
    x = pixels_per_degree_x * angle_from_vertical
    pixels_per_degree_y = permanent_camera_params.image_dim_y/permanent_camera_params.vertical_FoV
    y = pixels_per_degree_y * angle_from_horizontal

    return (x, y)
end

#returns the angle between the vector and the plane
#a,b,c is coefficients of plane. x, y, z is the vector.
function get_angle(a, b, c, x, y, z)
    numerator = a*x + b*y + c*z #took out abs
    denominator = sqrt(a^2+b^2+c^2) * sqrt(x^2+y^2+z^2)
    return asin(numerator/denominator) #angle in radians
end

#returns a,b,c for the vertical plane. only need the camera parameters
function get_vertical_plane(camera_params)
    p1 = camera_params.camera_location
    p2 = camera_params.camera_focus
    p3 = Coordinate(p1.x, p1.y, p1.z+1)
    return get_abc_plane(p1, p2, p3)
end

#need camera parameters and vertical plane (so horizontal will be perpendicular to it)
function get_horizontal_plane(camera_params, a, b, c)
    p1 = camera_params.camera_location
    p2 = camera_params.camera_focus
    p3 = Coordinate(p1.x+a, p1.y+b, p1.z+c) #adding normal vector (a, b, c) to the point to make a third point on the horizontal plane
    return get_abc_plane(p1, p2, p3)
end

#given 3 points on a plane (p1, p2, p3), get a, b, and c coefficients in the general form.
#abc also gives a normal vector
function get_abc_plane(p1::Coordinate, p2::Coordinate, p3::Coordinate)
    #using Method 2 from wikipedia: https://en.wikipedia.org/wiki/Plane_(geometry)#:~:text=In%20mathematics%2C%20a%20plane%20is,)%20and%20three%2Ddimensional%20space.
    D = det([p1.x p2.x p3.x; p1.y p2.y p3.y; p1.z p2.z p3.z])

    if D==0
        println("crap! determinant D=0")
    end
    #implicitly going to say d=-1 to obtain solution set
    a = det([1 1 1; p1.y p2.y p3.y; p1.z p2.z p3.z])/D
    b = det([p1.x p2.x p3.x; 1 1 1; p1.z p2.z p3.z])/D
    c = det([p1.x p2.x p3.x; p1.y p2.y p3.y; 1 1 1])/D

    return (a, b, c)
end
