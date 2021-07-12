#for running test cases from the generative model
using MetaGen
using Gen
using PyPlot
const plt = PyPlot
using Random

Random.seed!(1234)

################################################################################
num_videos = 30
num_frames = 100

#for testing purposes, let's fix V
cm = choicemap()
for v = 1:num_videos
	# cm[:videos => v => :v_matrix => :lambda_fa => 1 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 2 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 3 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 4 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 5 => :fa] = 0.1 #super high false alarm rate for category 5
	# cm[:videos => v => :v_matrix => :miss_rate => 1 => :miss] = 0.000001 #tiny miss rate for category 1
	# cm[:videos => v => :v_matrix => :miss_rate => 2 => :miss] = 0.01
	# cm[:videos => v => :v_matrix => :miss_rate => 3 => :miss] = 0.01
	# cm[:videos => v => :v_matrix => :miss_rate => 4 => :miss] = 0.99 #huge miss rate for category 4
	# cm[:videos => v => :v_matrix => :miss_rate => 5 => :miss] = 0.01
	for f = 1:num_frames
		camera_params = camera_trajectories[v, f]
		cm[:videos => v => :frame_chain => f => :camera => :camera_location_x] = camera_params.camera_location.x
		cm[:videos => v => :frame_chain => f => :camera => :camera_location_y] = camera_params.camera_location.y
		cm[:videos => v => :frame_chain => f => :camera => :camera_location_z] = camera_params.camera_location.z
		cm[:videos => v => :frame_chain => f => :camera => :camera_focus_x] = camera_params.camera_focus.x
		cm[:videos => v => :frame_chain => f => :camera => :camera_focus_y] = camera_params.camera_focus.y
		cm[:videos => v => :frame_chain => f => :camera => :camera_focus_z] = camera_params.camera_focus.z

		#for rf = 1:num_receptive_fields
			#println("objects_observed[v, f][rf] ", objects_observed[v, f][rf])
			#println("type ", typeof(objects_observed[v, f][rf]))
		cm[:videos => v => :frame_chain => f => :observations_2D] = convert(Array{Any, 1}, objects_observed[v, f])
		#end
	end
	#def
end

gt_trace,_ = Gen.generate(main, (num_videos, num_frames), cm)
#println(gt_trace)
gt_choices = get_choices(gt_trace)

params = Video_Params()

gt_v = zeros(Float64, length(params.possible_objects), 2)
for j = 1:length(params.possible_objects)
	gt_v[j,1] = gt_choices[:videos => 1 => :v_matrix => :lambda_fa => j => :fa]
	gt_v[j,2] = gt_choices[:videos => 1 => :v_matrix => :miss_rate => j => :miss]
end

receptive_fields = make_receptive_fields()
num_receptive_fields = length(receptive_fields)

#might be a bug in getting the objects_observed right
objects_observed = Matrix{Array{Array{Detection2D}}}(undef, num_videos, num_frames)
camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)
for v= 1:num_videos
	for f = 1:num_frames
		temp = []
		for rec_field = 1:num_receptive_fields
			println(gt_choices[:videos => v => :frame_chain => f => :observations_2D => :observations_2D => rec_field])
			temp = vcat(temp, convert(Array{Detection2D}, gt_choices[:videos => v => :frame_chain => f => :observations_2D => :observations_2D => rec_field]))
		end
		temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
		objects_observed[v, f] = temp_sorted_into_rfs

		location_x = gt_choices[:videos => v => :frame_chain => f => :camera => :camera_location_x]
		location_y = gt_choices[:videos => v => :frame_chain => f => :camera => :camera_location_y]
		location_z = gt_choices[:videos => v => :frame_chain => f => :camera => :camera_location_z]
		focus_x = gt_choices[:videos => v => :frame_chain => f => :camera => :camera_focus_x]
		focus_y = gt_choices[:videos => v => :frame_chain => f => :camera => :camera_focus_y]
		focus_z = gt_choices[:videos => v => :frame_chain => f => :camera => :camera_focus_z]
		c = Camera_Params(camera_location = Coordinate(location_x, location_y, location_z), camera_focus = Coordinate(focus_x, focus_y, focus_z))
		camera_trajectories[v, f] = c
	end
end

println("objects_observed ", objects_observed)
println("camera_trajectories ", camera_trajectories)

#visualize_observations(objects_observed, 1, 1, receptive_fields)
#visualize_observations(objects_observed, 1, 2, receptive_fields)

outfile = string("test_case.csv")
file = open(outfile, "w")
#file header
print(file, "gt_V & ")
for v=1:29
	print(file, "avg V ", v, " & ")
    print(file, "dictionary realities PF for scene ", v, " & ")
	print(file, "mode realities PF for scene ", v, " & ")
end
print(file, "avg V 30 &")
print(file, "dictionary realities PF for scene 30 & ")
print(file, "mode realities PF for scene 30")

print(file, "\n")

print(file, gt_v, " & ")

num_particles = 10
traces = unfold_particle_filter(num_particles, objects_observed, camera_trajectories, num_receptive_fields, file)
println("done")
#visualize_observations(objects_observed, 1, 1, receptive_fields)
#visualize_trace(traces, 1, camera_trajectories, 1, 1, params)

close(file)

# Gets v_hat matrix
best_trace = Gen.get_choices(traces[1])
v_hat = Matrix{Real}(undef, length(params.possible_objects), 2)
for j=1:length(params.possible_objects)
	v_hat[j, 1] = best_trace[:videos => num_videos => :v_matrix => :lambda_fa => j => :fa]
	v_hat[j, 2] = best_trace[:videos => num_videos => :v_matrix => :miss_rate => j => :miss]
end

#plot errors of final estimate
L1_errors = vec(abs.(gt_v - v_hat))

plt.clf()
plt.title("V matrix L1 errors, $(num_videos) videos, $(num_frames) frames, $(num_particles) particles")
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.hist(L1_errors, bins=bins)
plt.xlim(0, 1)
plt.savefig("errors_$(num_videos)_$(num_frames)_$(num_particles).pdf")
plt.clf()

#plot MSE as function of videos
MSEs = []

for i=1:num_videos
	# makes v_hat
	v_pred = Matrix{Real}(undef, length(params.possible_objects), 2)
	for j=1:length(params.possible_objects)
		v_pred[j, 1] = best_trace[:videos => i => :v_matrix => :lambda_fa => j => :fa]
		v_pred[j, 2] = best_trace[:videos => i => :v_matrix => :miss_rate => j => :miss]
	end
	mse = sum((gt_v - v_pred).^2) / (2 * length(params.possible_objects))
	push!(MSEs, mse)
end

plt.plot(MSEs)
plt.title("MSEs $(num_videos) videos, $(num_frames) frames, $(num_particles) particles")
plt.savefig("mses_$(num_videos)_$(num_frames)_$(num_particles).pdf")
plt.clf()
