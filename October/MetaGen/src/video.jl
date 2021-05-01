
"""
    This function takes a category and params and it returns the possible
    objects (as Detections) of that category that could be detected
"""
#hallucinate objects in 2D image
@gen function gen_possible_hallucination(params::Video_Params, cat::Int64)
    x = @trace(uniform(0,params.image_dim_x), :x)
    y = @trace(uniform(0,params.image_dim_y), :y)

    return (x, y, cat)
end
possible_hallucination_map = Gen.Map(gen_possible_hallucination)

#given a 3D detection, return BernoulliElement over a 2D detection
function render(params::Video_Params, camera_params::Camera_Params, object_3D::Object3D)
    cat = object_3D[4]
    object = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    x, y = get_image_xy(camera_params, params, object)

    return (x, y, cat)
end


@gen function gen_camera(params::Video_Params)
    #camera location
    camera_location_x = @trace(uniform(params.x_min,params.x_max), :camera_location_x)
    camera_location_y = @trace(uniform(params.y_min,params.y_max), :camera_location_y)
    camera_location_z = @trace(uniform(params.z_min,params.z_max), :camera_location_z)

    #camera focus focus
    camera_focus_x = @trace(uniform(params.x_min,params.x_max), :camera_focus_x)
    camera_focus_y = @trace(uniform(params.y_min,params.y_max), :camera_focus_y)
    camera_focus_z = @trace(uniform(params.z_min,params.z_max), :camera_focus_z)

    camera_params = Camera_Params(Coordinate(camera_location_x,camera_location_y,camera_location_z), Coordinate(camera_focus_x,camera_focus_y,camera_focus_z))
end

#state is Array{Any,1}
@gen function frame_kernel(current_frame::Int64, state, params::Video_Params, receptive_fields::Vector{Receptive_Field})

    ####Update 2D real objects

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    n_real_objects = length(state)
    paramses = fill(params, n_real_objects)
    camera_paramses = fill(camera_params, n_real_objects)
    real_detections = map(render, paramses, camera_paramses, state)
    real_detections = Array{Detection2D}(real_detections)
    #observations_2D will be what we condition on

    rfs_vec = get_rfs_vec(receptive_fields, real_detections, params)

    #for loop over receptive fields
    #@show maximum(map(length, rfs_vec))
    #could re-write with map
    #@trace(map(rfs)(rfs_vec), :observations_2D)

    for i = 1:length(rfs_vec)
        observations_2D = @trace(rfs(rfs_vec[i]), (i => :observations_2D))
    end

    return state #just keep sending the scene / initial state in.
end

frame_chain = Gen.Unfold(frame_kernel)

@gen function video_kernel(num_frames::Int64, params::Video_Params, receptive_fields::Array{Receptive_Field, 1})
    rfs_element = PoissonElement{Object3D}(params.lambda_objects, object_distribution, (params,))
    rfs_element = RFSElements{Object3D}([rfs_element]) #need brackets because rfs has to take an array
    init_state = @trace(rfs(rfs_element), :init_scene)
    states = @trace(frame_chain(num_frames, init_state, params, receptive_fields), :frame_chain)
    return (init_state, states)
end

#video_chain = Gen.Unfold(video_kernel)
video_map = Gen.Map(video_kernel)

frame_chain = Gen.Unfold(frame_kernel)

export video_map
