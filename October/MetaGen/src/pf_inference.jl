
function unfold_particle_filter(num_particles::Int, objects_observed::Matrix{Array{Detection2D}})

    init_obs = Gen.choicemap()
    #no video, no frames
    state = Gen.initialize_particle_filter(metacog, (possible_objects, 0, 0), init_obs, num_particles)

    for v=1:num_videos

        #want to do this for every frame at once. put a : instead of f
        obs = Gen.choicemap([:videos => v => :frame_chain => : => observations_2D] = objects_observed[v, :])


        Gen.particle_filter_step!(state, (possible_objects, v, num_frames), (UnknownChange(),), obs)

    end

    return Gen.sample_unweighted_traces(state, num_particles)

end
