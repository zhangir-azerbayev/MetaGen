# """
#     gen_possible_hallucination(params::Video_Params, cat::Int64)
#
# This function takes a category and params and it returns the possible
# objects (as 2D Detections) of that category that could be detected
# """
# #hallucinate objects in 2D image
# @gen (static) function gen_possible_hallucination(params::Video_Params, cat::Int64)
#     x = @trace(uniform(0,params.image_dim_x), :x)
#     y = @trace(uniform(0,params.image_dim_y), :y)
#     return (x, y, cat)
# end
# possible_hallucination_map = Gen.Map(gen_possible_hallucination)

"""given a 3D detection, return BernoulliElement over a 2D detection"""
function render(params::Video_Params, camera_params::Camera_Params, object_3D::Object3D)
    cat = object_3D[4]
    object = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    x, y = get_image_xy(camera_params, params, object)
    return (x, y, cat)
end

"""
    gen_camera(params::Video_Params)

Independently samples a camera location and camera focus from a
uniform distribution
"""
@gen (static) function gen_camera(params::Video_Params)
    #camera location
    camera_location_x = @trace(uniform(params.x_min,params.x_max), :camera_location_x)
    camera_location_y = @trace(uniform(params.y_min,params.y_max), :camera_location_y)
    camera_location_z = @trace(uniform(params.z_min,params.z_max), :camera_location_z)

    #camera focus focus
    camera_focus_x = @trace(uniform(params.x_min,params.x_max), :camera_focus_x)
    camera_focus_y = @trace(uniform(params.y_min,params.y_max), :camera_focus_y)
    camera_focus_z = @trace(uniform(params.z_min,params.z_max), :camera_focus_z)

    camera_params = Camera_Params(Coordinate(camera_location_x,camera_location_y,camera_location_z), Coordinate(camera_focus_x,camera_focus_y,camera_focus_z))
    return camera_params
end

#the only function that's not static. couldn't get rfs to play nicely with Map
@gen function rfs_helper(rfs_vec::Any)#couldn't get the type quite right
    for i = 1:length(rfs_vec)
        observations_2D = @trace(rfs(rfs_vec[i]), :observations_2D => i)
    end
end

"""
Generates the next frame given the current frame.

state is Array{Any,1}
"""
@gen (static) function frame_kernel(current_frame::Int64, state, params::Video_Params, v::Matrix{Real}, receptive_fields::Vector{Receptive_Field})

    ####Update 2D real objects

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    n_real_objects = length(state)
    paramses = fill(params, n_real_objects)
    #vs = fill(v, n_real_objects)
    camera_paramses = fill(camera_params, n_real_objects)
    real_detections = map(render, paramses, camera_paramses, state)
    real_detections = Array{Detection2D}(real_detections)
    #observations_2D will be what we condition on

    rfs_vec = get_rfs_vec(receptive_fields, real_detections, params, v)

    #for loop over receptive fields
    #@show maximum(map(length, rfs_vec))
    #could re-write with map
    #@trace(Gen.Map(rfs)(rfs_vec), :observations_2D) #gets no method matching error
    observations_2D = @trace(rfs_helper(rfs_vec), :observations_2D)

    return state #just keep sending the scene / initial state in.
end

frame_chain = Gen.Unfold(frame_kernel)

"""
Samples new values for lambda_fa based on the previous.
"""
@gen (static) function update_lambda_fa(previous_lambda_fa::Real, t::Int64)
    sd = max(1/10 - (t/1000), 1/1000)
    fa = @trace(trunc_normal(previous_lambda_fa, sd, 0.0, 100000.0), :fa)
    return fa
end

"""
Samples new values for miss_rate based on the previous.
"""
@gen (static) function update_miss_rate(previous_miss_rate::Real, t::Int64)
    sd = max(1 - (t/100), 1/100)
    miss = @trace(trunc_normal(previous_miss_rate, sd, 0.0, 1.0), :miss)
    return miss
end

"""
Samples a new v based on the previous v.
"""
@gen (static) function update_v_matrix(previous_v_matrix::Matrix{Real}, t::Int64)
    #v = Matrix{Real}(undef, dim(previous_v_matrix))
    ts = fill(t, length(previous_v_matrix[:,1]))
    fa = @trace(Map(update_lambda_fa)(previous_v_matrix[:,1], ts), :lambda_fa)
    miss = @trace(Map(update_miss_rate)(previous_v_matrix[:,2], ts), :miss_rate)
    #v[:, 1] = fa
    #v[:, 2] = miss
    v = hcat(fa, miss)
    return convert(Matrix{Real}, v)
    #return v
end

"""
Samples a new scene and a new v_matrix.
"""
@gen (static) function video_kernel(current_video::Int64, previous_v_matrix::Matrix{Real}, num_frames::Int64, params::Video_Params, receptive_fields::Array{Receptive_Field, 1})
    #for the scene. scenes are completely independent of each other
    rfs_element = GeometricElement{Object3D}(params.p_objects, object_distribution, (params,))
    rfs_element = RFSElements{Object3D}([rfs_element]) #need brackets because rfs has to take an array
    init_state = @trace(rfs(rfs_element), :init_scene)

    #for the metacognition.
    v_matrix = @trace(update_v_matrix(previous_v_matrix, current_video), :v_matrix)

    #make the observations
    states = @trace(frame_chain(num_frames, init_state, params, v_matrix, receptive_fields), :frame_chain)
    return v_matrix
end

#video_chain = Gen.Unfold(video_kernel)
"""Creates scene chain"""
video_chain = Gen.Unfold(video_kernel)

"""Creates frame chain"""
frame_chain = Gen.Unfold(frame_kernel)

export video_chain
