using MetaGen
using Gen
using PyPlot
const plt = PyPlot
using Random

Random.seed!(1234);

# Generates a v matrix
params = Video_Params()
v = make_visual_system(params)
print(v)

# There's just one receptive field
receptive_fields = make_receptive_fields()
num_receptive_fields = length(receptive_fields)

# Generates scenes and detections.
num_videos = 2
num_frames = 2


fs = fill(num_frames, num_videos) #number of frames per video
ps = fill(params, num_videos)
vs = fill(v, num_videos)
receptive_fieldses = fill(receptive_fields, num_videos)

Gen.load_generated_functions()
scenes_trace = Gen.simulate(video_map, (fs, ps, vs, receptive_fieldses))

#Gen.get_choices(scenes_trace)


#converts trace to arrays required for `unfold_particle_filter`
objects_observed = Matrix{Array{Array{Detection2D}}}(undef, num_videos, num_frames)
camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)

for v=1:num_videos
    for f=1:num_frames
        temp = scenes_trace[v => :frame_chain => f => 1 => :observations_2D]
	    temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
	    objects_observed[v, f] = temp_sorted_into_rfs

        camera_trajectories[v, f] = scenes_trace[v => :frame_chain => f => :camera]
    end
end

# Does inference

num_particles = 1
inference_traces = unfold_particle_filter(num_particles, objects_observed, camera_trajectories, num_receptive_fields)

# Gets best particle
top_index = 0
top_score = -Inf
for (i, trace) in enumerate(inference_traces)
    score = get_score(trace)
    if score > top_score
        global top_score = score
        global top_index = i
    end
end

best_trace = inference_traces[top_index]

# evaluates inferred visual_system.
v_hat = Matrix{Real}(undef, length(params.possible_objects), 2)

#Gen.get_choices(best_trace)
for j=1:length(params.possible_objects)
    v_hat[j, 1] = best_trace[:v_matrix => (:lambda_fa, j)]
    v_hat[j, 2] = best_trace[:v_matrix => (:miss_rate, j)]
end

L1_errors = vec(abs.(v - v_hat))

plt.clf()
plt.title("V matrix L1 errors")
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.hist(L1_errors, bins=bins)
plt.xlim(0, 1)
plt.savefig("errors.pdf")
