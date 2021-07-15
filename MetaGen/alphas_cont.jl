#trying piece-by-piece to build video.jl / break the working example
using Gen #to load the right GenRFS
using GenRFS
using Distributions

include("src/declaring_structs.jl")
include("src/custom_distributions.jl")
include("src/geometry_optics.jl")
include("src/inverse_optics.jl")
include("src/receptive_fields.jl")

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

    for i in setdiff(collect(1:num_cats), observations_2D_cats_edited)
        betas[i, 1] = betas[i, 1] + 1 #not hallucinated
    end
    return alphas, betas
end

function render(params::Video_Params, camera_params::Camera_Params, object_3D::Object3D)
    cat = object_3D[4]
    object = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    x, y = get_image_xy(camera_params, params, object)
    return (x, y, cat)
end

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

function update_alphas(alphas, s)
    alphas = alphas .+ s #update alpha
    return alphas
end

@gen (static) function update_lambda_fa(alpha::Int64, beta::Int64)
    fa = @trace(beta(alpha, beta), :fa)
    return fa
end

@gen (static) function update_miss_rate(alpha::Int64, beta::Int64)
    miss = @trace(beta(alpha, beta), :miss)
    return miss
end

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

@gen (static) function frame_kernel(iter::Int64, state, params::Video_Params, v::Matrix{Real}, receptive_fields::Vector{Receptive_Field})
    scene = state[1]
    alphas = state[2]
    betas = state[3]

    n_real_objects = length(scene)
    paramses = fill(params, n_real_objects)
    camera_params = @trace(gen_camera(params), :camera)
    camera_paramses = fill(camera_params, n_real_objects)
    real_detections = map(render, paramses, camera_paramses, scene)
    real_detections = Array{Detection2D}(real_detections)

    rfs_vec = get_rfs_vec(receptive_fields, real_detections, params, v)
    observations_2D = @trace(rfs(rfs_vec[1]), :observations_2D) #dirty shortcut because we only have one receptive field atm

    s = @trace(uniform_discrete(0, 10), :s)
    #alphas = update_alphas(alphas, s)

    alphas, betas = update_alpha_beta(alphas, betas, observations_2D, real_detections)
    new_state = (scene, alphas, betas)
    return new_state
end

frame_chain = Gen.Unfold(frame_kernel)

@gen function video_kernel(current_video::Int64, v_matrix_state::Any, num_frames::Int64, params::Video_Params, receptive_fields::Vector{Receptive_Field})
    #rfs_element = GeometricElement{Object3D}(params.p_objects, object_distribution, (params,))
    rfs_element = PoissonElement{Object3D}(params.p_objects, object_distribution, (params,))
    rfs_element = RFSElements{Object3D}([rfs_element]) #need brackets because rfs has to take an array
    init_scene = @trace(rfs(rfs_element), :init_scene)

    previous_v_matrix = v_matrix_state[1]
    previous_alphas = v_matrix_state[2]
    previous_betas = v_matrix_state[3]
    println("previous_alphas ", previous_alphas)
    println("previous_betas ", previous_betas)
    init_state = (init_scene, previous_alphas, previous_betas)

    #r = @trace(uniform(0.0, previous_alphas[1,1]), :r)
    #init_inner_state = (r, previous_alphas[1,1])
    #new_inner_state = @trace(inner_chain(num_frames, init_inner_state), :inner_chain)
    new_inner_state = @trace(frame_chain(num_frames, init_state, params, previous_v_matrix, receptive_fields), :frame_chain)
    #println("new_inner_state ", new_inner_state)
    new_alphas = previous_alphas
    alpha = new_inner_state[end][2]
    new_alphas = alpha
    v_matrix = @trace(update_v_matrix(new_alphas, previous_betas), :v_matrix)
    new_outer_state = (v_matrix, new_alphas, previous_betas)
    println("alpha out ", alpha)
    return new_outer_state
end

video_chain = Gen.Unfold(video_kernel)

@gen (static) function lambda_fa(arg1::Real, arg2::Real)
	fa = @trace(uniform(arg1, arg2), :fa)
	return fa
end

@gen (static) function miss_rate(arg1::Real, arg2::Real)
	miss = @trace(uniform(arg1, arg2), :miss)
	return miss
end

@gen (static) function make_visual_system(params::Video_Params)
	arg1 = fill(0.0, length(params.possible_objects))
	arg2 = fill(1.0, length(params.possible_objects))
	fa = @trace(Map(lambda_fa)(arg1, arg2), :lambda_fa) #these are lambdas per receptive field
	miss = @trace(Map(miss_rate)(arg1, arg2), :miss_rate)
	v = hcat(fa, miss)
   	return convert(Matrix{Real}, v)
end

#@gen (static) function main()
@gen (static) function main_fxn(num_videos::Int64, num_frames::Int64)
    params = Video_Params()
    receptive_fields = make_receptive_fields()
    init_v_matrix = @trace(make_visual_system(params), :init_v_matrix)
    alphas = fill(1, size(init_v_matrix))
	betas = fill(1, size(init_v_matrix))
	v_matrix_state = (init_v_matrix, alphas, betas)
	@trace(video_chain(num_videos, v_matrix_state, num_frames, params, receptive_fields), :videos)
end

@load_generated_functions

cm = choicemap()
cm[:videos => 5 => :inner_chain => 3 => :s] = 1000
gt_trace,_ = Gen.generate(main_fxn, (5, 3), cm);
#get_choices(gt_trace)

#new_trace,_ = regenerate(gt_trace, select(:videos => 4 => :inner_chain => 2 => :s));
#works when main is not static
#when main is static, doesn't print anything
#get_choices(new_trace)
