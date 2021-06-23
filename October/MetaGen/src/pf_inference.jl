"""
    unfold_particle_filter(num_particles::Int, objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params}, num_receptive_fields::Int64)

Performs inference procedure.

# Arguments
- num_particles::Int
- objects_observed: *how are these indexed?*
- camera_trajectories: *how are these indexed?*
- num_receptive_fields::Int64

array for each frame and array for each receptive field and array for those detections

# Returns
- A trace containing the inferred visual system and world state. 


"""
function unfold_particle_filter(num_particles::Int, objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params}, num_receptive_fields::Int64)
    init_obs = Gen.choicemap()

    #params set to default
    params = Video_Params()

    #for degugging purposes, condition on V
    for j = 1:length(params.possible_objects)
		#set lambda when target absent
		#v[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to fa rate of around 0.1
		init_obs[:v_matrix => (:lambda_fa, j)] = 0.002 #these are lambdas per receptive field
		#set miss rate when target present
		init_obs[:v_matrix => (:miss_rate, j)] = 0.25
	end

    #no video, no frames
    state = Gen.initialize_particle_filter(metacog, (0, 0), init_obs, num_particles)

    num_videos, num_frames = size(objects_observed)

    for v=1:num_videos

        println("v ", v)

        maybe_resample!(state, ess_threshold=num_particles)#used to be /2. now always resampling becasue I want to get rid of -Inf before they become NANs
        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after resample ", ess)


        #want to do this for every frame at once. put a : instead of f
        #rf is for receptive field
        obs = Gen.choicemap()

        for f = 1:num_frames
            camera_params = camera_trajectories[v, f]
            obs[:videos => v => :frame_chain => f => :camera => :camera_location_x] = camera_params.camera_location.x
            obs[:videos => v => :frame_chain => f => :camera => :camera_location_y] = camera_params.camera_location.y
            obs[:videos => v => :frame_chain => f => :camera => :camera_location_z] = camera_params.camera_location.z
            obs[:videos => v => :frame_chain => f => :camera => :camera_focus_x] = camera_params.camera_focus.x
            obs[:videos => v => :frame_chain => f => :camera => :camera_focus_y] = camera_params.camera_focus.y
            obs[:videos => v => :frame_chain => f => :camera => :camera_focus_z] = camera_params.camera_focus.z

            for rf = 1:num_receptive_fields
                #println("objects_observed[v, f][rf] ", objects_observed[v, f][rf])
                #println("type ", typeof(objects_observed[v, f][rf]))
                obs[:videos => v => :frame_chain => f => rf => :observations_2D] = convert(Array{Any, 1}, objects_observed[v, f][rf])
            end
        end
        #def should be using map to replace for loops here
        #point is, condition on the camera trajectory and on the observations

        Gen.particle_filter_step!(state, (v, num_frames), (UnknownChange(),), obs)

        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after pf step ", ess)
        # for i = 1:num_particles
        #     println("weight ", state.log_weights[i])
        # #     println("frame 1")
        # #     println(state.traces[i][:videos => v => :frame_chain => 1 => :camera])
        # end

        #optional rejuvination
        (perturb_params, n_objects_per_category) = get_probs_categories(objects_observed, params, v, num_frames, num_receptive_fields)
        line_segments_per_category = get_line_segments_per_category(params, objects_observed, camera_trajectories, v, num_frames, num_receptive_fields)
        #line_segments = get_line_segments(objects_observed, camera_trajectories, params, v, num_frames, num_receptive_fields, total_n_objects)
        println(line_segments_per_category)
        for i = 1:num_particles
            state.traces[i] = perturb_scene(state.traces[i], v, perturb_params, line_segments_per_category)
            println("done perturbing i ", i)
            println("trace ", state.traces[i][:videos => v => :init_scene])
            println("log score of this trace ", get_score(state.traces[i]))
            visualize_trace(state.traces, i, camera_trajectories, v, 1, params)
        end

        min_particle_weight = minimum(map(i -> get_score(state.traces[i]), collect(1:num_particles)))
        println("min log score ", min_particle_weight)

        max_particle_weight = maximum(map(i -> get_score(state.traces[i]), collect(1:num_particles)))
        println("max log score ", max_particle_weight)

        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after rejuvination ", ess)

    end

    return Gen.sample_unweighted_traces(state, num_particles)

end

"""
    perturb_scene(trace, v::Int64, perturb_params::Perturb_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})

Does HMC steps.
"""
@gen function perturb_scene(trace, v::Int64, perturb_params::Perturb_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
    acceptance_counter = 0

    for iter=1:500 #try 100 MH moves
        println("iter ", iter)
        println("trace ", trace[:videos => v => :init_scene])

        trace, accepted = add_remove_kernel(trace, v, line_segments_per_category, perturb_params)
        println("accepted? ", accepted)
        println("trace ", trace[:videos => v => :init_scene])

        acceptance_counter += accepted

        #only try changing location or category if there's at least one object in the scene
        if length(trace[:videos => v => :init_scene]) > 0
            trace, accepted = change_location_kernel(trace, v, 10.0, perturb_params)
            println("accepted? ", accepted)
            println("trace ", trace[:videos => v => :init_scene])
            acceptance_counter += accepted

            trace, accepted = change_category_kernel(trace, v, perturb_params)
            println("accepted? ", accepted)
            println("trace ", trace[:videos => v => :init_scene])
            acceptance_counter += accepted
        end
    end
    #println("acceptance_counter $(acceptance_counter/1500)")

    return trace
end

"""
    get_probs_categories(objects_observed::Matrix{Array{Array{Detection2D}}}, params::Video_Params, v::Int64, num_frames::Int64, num_receptive_fields::Int64)

Returns a probability distribution over the object categories to be used
in the proposal functions.

Derived from the objects that the visual system observes.
"""
function get_probs_categories(objects_observed::Matrix{Array{Array{Detection2D}}}, params::Video_Params, v::Int64, num_frames::Int64, num_receptive_fields::Int64)
    track_categories = zeros(length(params.possible_objects)) #each element will be the number of times that category was detected. adding 1
    for f = 1:num_frames
        for rf = 1:num_receptive_fields
            for (index, value) in enumerate(objects_observed[v, f][rf])
                track_categories[value[3]] = track_categories[value[3]]+1#category
            end
        end
    end
    return (Perturb_Params(probs_possible_objects = (track_categories.+1)./sum(track_categories.+1)), track_categories)
end

"""
    get_line_segments_per_category(params::Video_Params, objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params}, v::Int64, num_frames::Int64, num_receptive_fields::Int64)

Gets all line segments in 3d space that correspond to each 2D detection
in every frame of a scene.
"""
function get_line_segments_per_category(params::Video_Params, objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params}, v::Int64, num_frames::Int64, num_receptive_fields::Int64)
    line_segments = Array{Array{Line_Segment, 1}}(undef, length(params.possible_objects))
    for j = 1:length(params.possible_objects)
        line_segments[j] = []
    end
    for f = 1:num_frames
        camera_params = camera_trajectories[v, f]
        for rf = 1:num_receptive_fields
            for (index, value) in enumerate(objects_observed[v, f][rf])
                line_segment = get_line_segment(camera_params, params, value)
                push!(line_segments[value[3]], line_segment)
            end
        end
    end
    return line_segments
end

"""Duplicated from Gen library"""
function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

"""Duplicated from Gen Library"""
function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

export unfold_particle_filter
