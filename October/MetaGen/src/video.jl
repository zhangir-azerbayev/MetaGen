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

Frame = Vector{Detection}

"""
    This function takes a category and params and it returns the possible
    objects (as Detections) of that category that could be detected
"""
@gen function gen_possible_detection(params::Video_Params, cat::Int64)
    x = @trace(uniform(0,params.x_max), :x)
    y = @trace(uniform(0,params.y_max), :y)
    z = @trace(uniform(0,params.z_max), :z)

    println("params.v[:,2] ", params.v[:,2])
    hit = params.v[cat,2]

    sd_x = 1.
    sd_y = 1.
    sd_z = 1.
    cov = [sd_x 0. 0.; 0. sd_y 0.; 0. 0. sd_z]

    BernoulliElement{Detection}(hit, object_distribution_present,
                              ([x,y,z], cov, cat))
end

@gen function gen_possible_hallucination(params::Video_Params, cat::Int64)
    x = @trace(uniform(0,params.x_max), :x)
    y = @trace(uniform(0,params.y_max), :y)
    z = @trace(uniform(0,params.z_max), :z)

    println("params.v[:,1] ", params.v[:,1])
    fa = params.v[cat,1]

    sd_x = 1.
    sd_y = 1.
    sd_z = 1.
    cov = [sd_x 0. 0.; 0. sd_y 0.; 0. 0. sd_z]

    BernoulliElement{Detection}(fa, object_distribution_present,
                              ([x,y,z], cov, cat))
end

possible_detection_map = Gen.Map(gen_possible_detection)

possible_hallucination_map = Gen.Map(gen_possible_hallucination)

#given a 3D detection, return either a 2D detctection or NAs/nothing
@gen function gen_render(camera_params::Camera_Params, permanent_camera_params::Permanent_Camera_Params, observation_3D::Detection)
    object = Coordinate(observation_3D[1], observation_3D[2], observation_3D[3])
    x, y = get_image_xy(camera_params, permanent_camera_params, object)
    #add noise to this x and y
    sd_x = 1.
    sd_y = 1.
    sd_z = 1.
    cov = [sd_x 0. 0.; 0. sd_y 0.; 0. 0. sd_z]
    detection = @trace(object_distribution_image([x, y], cov, observation_3D[4]), :detection2d)
    #might cause a control flow problem
    #want to "crop image." so only return the detection if it's within the image dim
    if abs(detection[1])<(permanent_camera_params.image_dim_x/2) & abs(detection[2])<(permanent_camera_params.image_dim_y/2)
        return detection
    else
        return nothing
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
    @trace(possible_detection_map(paramses, c), :init_scene)
end

@gen function frame_kernel(current_frame::Int64, state, params)
    println("frame_kernel")
    ####Update imaginary objects

    println("params.v[:,1] ", params.v[:,1])
    sum_fas_imaginary_objects = sum(params.v[:,1])#get lambdas for absent
    numObjects = @trace(poisson(sum_fas_imaginary_objects), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #numObjects = @trace(poisson(2), (:numObjects)) #may want to truncate so 0 objects isn't possible

    #normalizing
    fas_normalized = params.v[:,1]./sum_fas_imaginary_objects
    println("fas_normalized ", fas_normalized)
    c = @trace(multinomial(numObjects, fas_normalized), (:c))

    paramses = fill(params, numObjects)
    imagined_objects = @trace(possible_hallucination_map(paramses, c), :imagined_objects)

    ####Actually render the 3D "observations" -- what might possibly be detected

    combined_obj = vcat(state, imagined_objects)
    display(params.v)
    println("length(state) ", length(state))
    println("length(imagined_objects) ", length(imagined_objects))
    #combined_obj = collect(PoissonElement{Detection}, combined_obj)
    combined_obj = RFSElements{Detection}(combined_obj)
    println("length(combined_obj) ", length(combined_obj))

    # n = length(combined_obj)
    # #many_element_rfs = collect()
    # #many_element_rfs = collect(Int64, map(i -> rand(Distributions.Categorical(probs)), 1:n))
    # many_element_rfs = RFSElements{Detection}(undef,n)
    # for i = 1:n
    #     many_element_rfs[i] = combined_obj[i]
    # end
    #observations = @trace(rfs(many_element_rfs), :observations)
    observations_3D = @trace(rfs(combined_obj), :observations_3D)

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    n_observations = length(observations_3D)
    camera_paramses = fill(camera_params, n_observations)
    permanent_camera_paramses = fill(permanent_camera_paramses, n_observations)
    observations_2D = @trace(render_map(camera_paramses, permanent_camera_paramses, observations_3D), :observations_2D)
    #observations_2D will be what we condition on

    return state #just keep sending the scene / initial state in.
end

frame_chain = Gen.Unfold(frame_kernel)

@gen function video_kernel(num_frames::Int64, params::Video_Params)

    println("in video kernel")

    init_state = @trace(init_scene(params), :init_state)
    states = @trace(frame_chain(num_frames, init_state, params), :frame_chain)
    return (init_state, states)
end

#video_chain = Gen.Unfold(video_kernel)
video_map = Gen.Map(video_kernel)

frame_chain = Gen.Unfold(frame_kernel)

export video_map, Video_Params
