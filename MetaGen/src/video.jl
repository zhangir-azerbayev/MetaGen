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

function within_frame(p::Detection2D)
    p[1] >= 0 && p[1] <= 320 && p[2] >= 0 && p[2] <= 240 #hard-codded frame size
end

#first approximation
function update_alpha_beta(alphas_old::Matrix{Int64}, betas_old::Matrix{Int64}, observations_2D, real_detections::Array{Detection2D})
    alphas = deepcopy(alphas_old)
    betas = deepcopy(betas_old)

    observations_2D = filter!(within_frame, observations_2D)
    real_detections = filter!(within_frame, real_detections)

    #lets only do this by category
    real_detections_cats = last.(real_detections)
    observations_2D_cats = last.(observations_2D)

    #see if actually detected. update miss rate
    observations_2D_cats_edited = copy(observations_2D_cats)
    for i = 1:length(real_detections_cats)
        current_cat = real_detections_cats[i]
        if current_cat in observations_2D_cats_edited #actually detected
            betas[current_cat, 2] = betas[current_cat, 2] + 1 #increment
            #remove
            observations_2D_cats_edited = deleteat!(observations_2D_cats_edited, findfirst(observations_2D_cats_edited.==current_cat))
        else #missed
            alphas[current_cat, 2] = alphas[current_cat, 2] + 1
        end
    end

    #everything left in observations_2D_cats_edited must have been hallucinated
    num_cats = size(alphas)[1]

    for i in 1:length(observations_2D_cats_edited)
        alphas[observations_2D_cats_edited[i], 1] = alphas[observations_2D_cats_edited[i], 1] + 1
    end

    #update beta for hallucinations
    betas[:, 1] = betas[:, 1] .+ 1 #add one for each frame

    return alphas, betas
end

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

"""
Generates the next frame given the current frame.

state is Tuple{Array{Any,1}, Matrix{Int64}, Matrix{Int64}}
"""
@gen (static) function frame_kernel(current_frame::Int64, state::Any, params::Video_Params, v::Matrix{Real}, receptive_fields::Vector{Receptive_Field})
    ####Update 2D real objects

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    scene = state[1]
    n_real_objects = length(scene)
    paramses = fill(params, n_real_objects)
    #vs = fill(v, n_real_objects)
    camera_paramses = fill(camera_params, n_real_objects)
    real_detections = map(render, paramses, camera_paramses, scene)
    real_detections = Array{Detection2D}(real_detections)
    #observations_2D will be what we condition on

    rfs_vec = get_rfs_vec(real_detections, params, v)

    #for loop over receptive fields
    #@show maximum(map(length, rfs_vec))
    #could re-write with map
    #@trace(Gen.Map(rfs)(rfs_vec), :observations_2D) #gets no method matching error
    observations_2D = @trace(rfs(rfs_vec), :observations_2D) #dirty shortcut because we only have one receptive field atm
    alphas, betas = update_alpha_beta(state[2], state[3], observations_2D, real_detections)
    state = (scene, alphas, betas) #just keep sending the scene in.
    return state
end

frame_chain = Gen.Unfold(frame_kernel)

"""
Samples new values for lambda_fa.
"""
@gen (static) function update_lambda_fa(alpha::Int64, beta::Int64)
    fa = @trace(gamma(alpha, 1/beta), :fa)
    return fa
end

"""
Samples new values for miss_rate.
"""
@gen (static) function update_miss_rate(alpha::Int64, beta::Int64)
    miss = @trace(beta(alpha, beta), :miss)
    return miss
end

"""
Samples a new v based on the previous v.
"""
@gen (static) function update_v_matrix(alphas::Matrix{Int64}, betas::Matrix{Int64})
    #v = Matrix{Real}(undef, dim(previous_v_matrix))
    fa = @trace(Map(update_lambda_fa)(alphas[:,1], betas[:,1]), :lambda_fa)
    miss = @trace(Map(update_miss_rate)(alphas[:,2], betas[:,2]), :miss_rate)
    #v[:, 1] = fa
    #v[:, 2] = miss
    v = hcat(fa, miss)
    return convert(Matrix{Real}, v)
    #return v
end

"""
Samples a new scene and a new v_matrix.
"""
@gen (static) function video_kernel(current_video::Int64, v_matrix_state::Any, num_frames::Int64, params::Video_Params, receptive_fields::Array{Receptive_Field, 1})
    #for the scene. scenes are completely independent of each other
    #println("current video ", current_video)

    #rfs_element = GeometricElement{Object3D}(params.p_objects, object_distribution, (params,))
    rfs_element = GeometricElement{Object3D}(params.p_objects, object_distribution, (params,))
    rfs_element = RFSElements{Object3D}([rfs_element]) #need brackets because rfs has to take an array
    init_scene = @trace(rfs(rfs_element), :init_scene)
    #make the observations
    previous_v_matrix = v_matrix_state[1]
    previous_alphas = v_matrix_state[2]
    previous_betas = v_matrix_state[3]
    init_state = (init_scene, previous_alphas, previous_betas)

    state = @trace(frame_chain(num_frames, init_state, params, previous_v_matrix, receptive_fields), :frame_chain)
    alphas = state[end][2]#not sure num_frames or end is better for index
    betas = state[end][3]
    #for the metacognition.
    v_matrix = @trace(update_v_matrix(alphas, betas), :v_matrix)
    v_matrix_state = (v_matrix, alphas, betas)
    return v_matrix_state
end

#video_chain = Gen.Unfold(video_kernel)
"""Creates scene chain"""
video_chain = Gen.Unfold(video_kernel)

"""Creates frame chain"""
frame_chain = Gen.Unfold(frame_kernel)

export video_chain
