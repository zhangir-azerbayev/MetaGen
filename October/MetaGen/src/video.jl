
"""
    This function takes a category and params and it returns the possible
    objects (as Detections) of that category that could be detected
"""
@gen function gen_location(params::Video_Params, cat::Int64)
    x = @trace(uniform(0,params.x_max), :x)
    y = @trace(uniform(0,params.y_max), :y)
    z = @trace(uniform(0,params.z_max), :z)

    return (x, y, z, cat) #want type Detection3D
end

#hallucinate objects in 2D image
@gen function gen_possible_hallucination(params::Video_Params, permanent_camera_params::Permanent_Camera_Params, cat::Int64)
    x = @trace(uniform(0,permanent_camera_params.image_dim_x), :x)
    y = @trace(uniform(0,permanent_camera_params.image_dim_y), :y)

    return (x, y, cat)
end

location_map = Gen.Map(gen_location)

possible_hallucination_map = Gen.Map(gen_possible_hallucination)

#given a 3D detection, return BernoulliElement over a 2D detection
function render(params::Video_Params, camera_params::Camera_Params, permanent_camera_params::Permanent_Camera_Params, object_3D::Detection3D)
    cat = object_3D[4]
    object = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    x, y = get_image_xy(camera_params, permanent_camera_params, object)

    return (x, y, cat)
end


@gen function gen_camera(params::Video_Params)
    #camera location
    camera_location_x = @trace(uniform(0,params.x_max), :camera_location_x)
    camera_location_y = @trace(uniform(0,params.y_max), :camera_location_y)
    camera_location_z = @trace(uniform(0,params.z_max), :camera_location_z)

    #camera focus focus
    camera_focus_x = @trace(uniform(0,params.x_max), :camera_focus_x)
    camera_focus_y = @trace(uniform(0,params.y_max), :camera_focus_y)
    camera_focus_z = @trace(uniform(0,params.z_max), :camera_focus_z)

    camera_params = Camera_Params(Coordinate(camera_location_x,camera_location_y,camera_location_z), Coordinate(camera_focus_x,camera_focus_y,camera_focus_z))
end

@gen function init_scene(params::Video_Params)
    #set up the scene / T0 world state

    #saying there must be at least one object per scene, and at most 100
    numObjects = @trace(trunc_poisson(20.0, 0.0, 100.0), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #objects = @trace(multinomial_objects(numObjects,[0.2,0.2,0.2,0.2,0.2], ), (:objects))
    c = @trace(multinomial(numObjects,[0.2,0.2,0.2,0.2,0.2], ), (:c))

    paramses = fill(params, numObjects)
    @trace(location_map(paramses, c), :locations) #add location
end

@gen function frame_kernel(current_frame::Int64, state, params::Video_Params, permanent_camera_params::Permanent_Camera_Params, receptive_fields::Vector{Receptive_Field})

    ####Update imaginary 2D objects

    sum_fas_imaginary_objects = sum(params.v[:,1])#get lambdas for absent
    numObjects = @trace(poisson(sum_fas_imaginary_objects), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #numObjects = @trace(poisson(2), (:numObjects)) #may want to truncate so 0 objects isn't possible

    #normalizing
    fas_normalized = params.v[:,1]./sum_fas_imaginary_objects
    c = @trace(multinomial(numObjects, fas_normalized), (:c))

    paramses = fill(params, numObjects)
    permanent_camera_paramses = fill(permanent_camera_params, numObjects)
    imaginary_detections = @trace(possible_hallucination_map(paramses, permanent_camera_paramses, c), :imagined_objects)
    imaginary_detections = Array{Detection2D}(imaginary_detections) #force it to the right type
    ####Update 2D real objects

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    n_real_objects = length(state)
    paramses = fill(params, n_real_objects)
    camera_paramses = fill(camera_params, n_real_objects)
    permanent_camera_paramses = fill(permanent_camera_params, n_real_objects)
    real_detections = map(render, paramses, camera_paramses, permanent_camera_paramses, state)
    real_detections = Array{Detection2D}(real_detections)
    #observations_2D will be what we condition on

    #points, categories -> detections
    rfs_vec = get_rfs_vec(receptive_fields, imaginary_detections, real_detections, params)


    #for loop over receptive fields
    for i = 1:length(rfs_vec)
        observations_2D = @trace(rfs(rfs_vec[i]), (i => :observations_2D))
    end


    return state #just keep sending the scene / initial state in.
end

frame_chain = Gen.Unfold(frame_kernel)

@gen function video_kernel(num_frames::Int64, params::Video_Params, permanent_camera_params::Permanent_Camera_Params, receptive_fields::Vector{Receptive_Field})

    init_state = @trace(init_scene(params), :init_state)
    states = @trace(frame_chain(num_frames, init_state, params, permanent_camera_params, receptive_fields), :frame_chain)
    return (init_state, states)
end

#video_chain = Gen.Unfold(video_kernel)
video_map = Gen.Map(video_kernel)

frame_chain = Gen.Unfold(frame_kernel)

export video_map, Video_Params
