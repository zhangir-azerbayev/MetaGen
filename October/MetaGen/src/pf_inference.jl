
#array for each frame and array for each receptive field and array for those detections
function unfold_particle_filter(num_particles::Int, objects_observed::Matrix{Array{Array{Detection2D}}}, camera_trajectories::Matrix{Camera_Params})

    init_obs = Gen.choicemap()

    #no video, no frames
    possible_objects = [1, 2, 3, 4, 5]
    state = Gen.initialize_particle_filter(metacog, (possible_objects, 0, 0), init_obs, num_particles)

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


            #num_receptive_fields should be a constant but whatever
            num_receptive_fields = length(objects_observed[v, f])
            for rf = 1:num_receptive_fields
                #println("objects_observed[v, f][rf] ", objects_observed[v, f][rf])
                #println("type ", typeof(objects_observed[v, f][rf]))
                obs[:videos => v => :frame_chain => f => rf => :observations_2D] = convert(Array{Any, 1}, objects_observed[v, f][rf])
            end
        end
        #def should be using map to replace for loops here
        #point is, condition on the camera trajectory and on the observations

        Gen.particle_filter_step!(state, (possible_objects, v, num_frames), (UnknownChange(),), obs)

        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after pf step ", ess)
        for i = 1:num_particles
            println("weight ", state.log_weights[i])
        #     println("frame 1")
        #     println(state.traces[i][:videos => v => :frame_chain => 1 => :camera])
        #     println("frame 2")
        #     println(state.traces[i][:videos => v => :frame_chain => 2 => :camera])
        #     println("frame 3")
        #     println(state.traces[i][:videos => v => :frame_chain => 3 => :camera])
        #     println("frame 4")
        #     println(state.traces[i][:videos => v => :frame_chain => 4 => :camera])
        #     println("frame 5")
        #     println(state.traces[i][:videos => v => :frame_chain => 5 => :camera])
        #     println("frame 6")
        #     println(state.traces[i][:videos => v => :frame_chain => 6 => :camera])
        end

        #later add stuff to not change numbers so far back

    end

    return Gen.sample_unweighted_traces(state, num_particles)

end

# function init_scene_proposal(trace, v)
#     choice_map = get_choices(trace)
#     es = choice_map[:videos => v => :init_scene]
#
#     n = length(es)
#     index = categorical(fill(1/n, n))
#     others = setdiff(1:n, index)
#     changed = helper(es[index], 0.5, 10.) #maybe remove
#     same = map(e -> helper(e, 1.0, 0.), es[others])#keep others the same
#
#     #maybe add
#     to_add = BernoulliElement{Detection2D}(0.5, object_distribution, (params,))
#
#     vec = cat(changed, same, to_add)
#
#     @trace(rfs(vec), (:videos => v => :init_scene))
#
#     #need to change downstream
# end

#
function helper(to_change, bernoulli_p, variance)
    cov = diagm([variance, variance, variance]) #might have to worry about going outside of the world's dimensions
    BernoulliElement{Detection2D}(bernoulli_p, object_distribution_present, ([to_change[1], to_change[2], to_change[3]], cov, to_change[4]))
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
