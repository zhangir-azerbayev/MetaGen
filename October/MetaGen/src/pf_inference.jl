
#array for each frame and array for each receptive field and array for those detections
function unfold_particle_filter(num_particles::Int, objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params}, num_receptive_fields::Int64)
    init_obs = Gen.choicemap()

    #no video, no frames
    state = Gen.initialize_particle_filter(metacog, (0, 0), init_obs, num_particles)

    #num_videos, num_frames = size(objects_observed)
    num_videos = 1 #10
    num_frames = 75

    #params set to default
    params = Video_Params()

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
        (perturb_params, total_n_objects) = get_probs_categories(objects_observed, params, v, num_frames, num_receptive_fields)
        line_segments = get_line_segments(objects_observed, camera_trajectories, params, v, num_frames, num_receptive_fields, total_n_objects)
        for i = 1:num_particles
            state.traces[i] = perturb_scene(state.traces[i], v, perturb_params, line_segments)
            println("done perturbing i ", i)
        end

        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after rejuvination ", ess)

    end

    return Gen.sample_unweighted_traces(state, num_particles)

end

@gen function perturb_scene(trace, v::Int64, perturb_params::Perturb_Params, line_segments::Array{Line_Segment})
    for iter=1:100 #try 100 MH moves
        println("iter ", iter)

        trace, accepted = add_remove_kernel(trace, v, line_segments, perturb_params)
        println("accepted? ", accepted)
        println("trace ", trace[:videos => v => :init_scene])

        trace, accepted = change_location_kernel(trace, v, 10.0, perturb_params)
        println("accepted? ", accepted)
        println("trace ", trace[:videos => v => :init_scene])

        trace, accepted = change_category_kernel(trace, v, perturb_params)
        println("accepted? ", accepted)
        println("trace ", trace[:videos => v => :init_scene])

    end
    return trace
end

function get_probs_categories(objects_observed::Matrix{Array{Array{Detection2D}}}, params::Video_Params, v::Int64, num_frames::Int64, num_receptive_fields::Int64)
    track_categories = zeros(length(params.possible_objects)) #each element will be the number of times that category was detected. adding 1
    for f = 1:num_frames
        for rf = 1:num_receptive_fields
            for (index, value) in enumerate(objects_observed[v, f][rf])
                track_categories[value[3]] = track_categories[value[3]]+1#category
            end
        end
    end
    total_objects = convert(Int64, sum(track_categories))
    track_categories = track_categories.+0.01
    return (Perturb_Params(probs_possible_objects = track_categories./sum(track_categories)), total_objects)
end

function get_line_segments(objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params}, params::Video_Params, v::Int64, num_frames::Int64, num_receptive_fields::Int64, total_n_objects::Int64)
    line_segments = Array{Line_Segment, 1}(undef, total_n_objects) #each element will be the number of times that category was detected. adding 1
    i = 1
    for f = 1:num_frames
        camera_params = camera_trajectories[v, f]
        for rf = 1:num_receptive_fields
            for (index, value) in enumerate(objects_observed[v, f][rf])
                line_segment = get_line_segment(camera_params, params, value)
                line_segments[i] = line_segment
                i = i+1
            end
        end
    end
    return line_segments
end

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

export unfold_particle_filter
