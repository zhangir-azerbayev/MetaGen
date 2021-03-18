
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

    println("params.v[:,1] ", params.v[:,1])
    fa = params.v[cat,1]

    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.; 0. sd_y;]

    BernoulliElement{Detection2D}(fa, object_distribution_image,
                              ([x, y], cov, cat))
end

location_map = Gen.Map(gen_location)

possible_hallucination_map = Gen.Map(gen_possible_hallucination)

#given a 3D detection, return BernoulliElement over a 2D detection
@gen function gen_render(params::Video_Params, camera_params::Camera_Params, permanent_camera_params::Permanent_Camera_Params, object_3D::Detection3D)
    cat = object_3D[4]
    object = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    x, y = get_image_xy(camera_params, permanent_camera_params, object)

    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.; 0. sd_y;]

    if abs(x)>(permanent_camera_params.image_dim_x/2) || abs(y)>(permanent_camera_params.image_dim_y/2)
        return BernoulliElement{Detection2D}(0, object_distribution_image, ([x, y], cov, cat)) #0 because chances of detecting it are 0
    else
        println("params.v[:,2] ", params.v[:,2])
        hit = params.v[cat,2]
        return BernoulliElement{Detection2D}(1-hit, object_distribution_image, ([x, y], cov, cat)) #Detetion2D
    end
end

render_map = Gen.Map(gen_render)

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
    println("init_scene")
    #set up the scene / T0 world state

    #saying there must be at least one object per scene, and at most 100
    numObjects = @trace(trunc_poisson(1.0, 0.0, 100.0), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #objects = @trace(multinomial_objects(numObjects,[0.2,0.2,0.2,0.2,0.2], ), (:objects))
    c = @trace(multinomial(numObjects,[0.2,0.2,0.2,0.2,0.2], ), (:c))

    paramses = fill(params, numObjects)
    @trace(location_map(paramses, c), :locations) #add location
end

@gen function frame_kernel(current_frame::Int64, state, params::Video_Params, permanent_camera_params::Permanent_Camera_Params)
    println("frame_kernel")

    ####Update imaginary 2D objects

    println("params.v[:,1] ", params.v[:,1])
    sum_fas_imaginary_objects = sum(params.v[:,1])#get lambdas for absent
    numObjects = @trace(poisson(sum_fas_imaginary_objects), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #numObjects = @trace(poisson(2), (:numObjects)) #may want to truncate so 0 objects isn't possible

    #normalizing
    fas_normalized = params.v[:,1]./sum_fas_imaginary_objects
    println("fas_normalized ", fas_normalized)
    c = @trace(multinomial(numObjects, fas_normalized), (:c))

    paramses = fill(params, numObjects)
    permanent_camera_paramses = fill(permanent_camera_params, numObjects)
    imagined_objects_2D = @trace(possible_hallucination_map(paramses, permanent_camera_paramses, c), :imagined_objects)

    ####Update 2D real objects

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    n_real_objects = length(state)
    paramses = fill(params, n_real_objects)
    camera_paramses = fill(camera_params, n_real_objects)
    permanent_camera_paramses = fill(permanent_camera_params, n_real_objects)
    real_objects_2D = @trace(render_map(paramses, camera_paramses, permanent_camera_paramses, state), :render_map)
    #observations_2D will be what we condition on


    ####Actually use rfs to sample the 2D observations
    combined_obj = vcat(real_objects_2D, imagined_objects_2D)
    #combined_obj = collect(PoissonElement{Detection}, combined_obj)
    println("length(combined_obj) ", length(combined_obj))
    println("combined_obj ", combined_obj)
    combined_obj = RFSElements{Detection2D}(combined_obj)
    println("length(combined_obj) ", length(combined_obj))

    # n = length(combined_obj)
    # #many_element_rfs = collect()
    # #many_element_rfs = collect(Int64, map(i -> rand(Distributions.Categorical(probs)), 1:n))
    # many_element_rfs = RFSElements{Detection}(undef,n)
    # for i = 1:n
    #     many_element_rfs[i] = combined_obj[i]
    # end
    #observations = @trace(rfs(many_element_rfs), :observations)
    observations_2D = @trace(rfs(combined_obj), :observations_2D)


    return state #just keep sending the scene / initial state in.
end

frame_chain = Gen.Unfold(frame_kernel)

@gen function video_kernel(num_frames::Int64, params::Video_Params, permanent_camera_params::Permanent_Camera_Params)

    println("in video kernel")

    init_state = @trace(init_scene(params), :init_state)
    println("init_state ", init_state)
    states = @trace(frame_chain(num_frames, init_state, params, permanent_camera_params), :frame_chain)
    return (init_state, states)
end

#video_chain = Gen.Unfold(video_kernel)
video_map = Gen.Map(video_kernel)

frame_chain = Gen.Unfold(frame_kernel)

export video_map, Video_Params
