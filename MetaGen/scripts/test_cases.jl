#for running hand-constructed test cases

using MetaGen
using JSON

#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
params = Video_Params()

receptive_fields = make_receptive_fields()
num_receptive_fields = length(receptive_fields)

num_videos = 30
num_frames = 10

objects_observed = Matrix{Array{Detection2D}}(undef, num_videos, num_frames)
camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)
for v=1:num_videos
	for f = 1

		labels = [2, 5]
		xs = [50.1, 225.5]
		ys = [60.2, 173.1]
		temp = Array{Detection2D}(undef, length(labels))
		for i = 1:length(labels)
			label = labels[i]
			x = xs[i]
			y = ys[i]
			temp[i] = (x, y, label) #so now I have an array of detections
		end
		#turn that array of detections into an array of an array of detections sorted by receptive_field
		#temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
		objects_observed[v, f] = temp

		c = Camera_Params(camera_location = Coordinate(0.01,0.02,0.001), camera_focus = Coordinate(1.0,1.0,1.0))
		camera_trajectories[v, f] = c
	end

	#################################
	#frame 2
	for f = 2:10

		labels = [2]
		xs = [50.1]
		ys = [60.2]
		temp = Array{Detection2D}(undef, length(labels))
		for i = 1:length(labels)
			label = labels[i]
			x = xs[i]
			y = ys[i]
			temp[i] = (x, y, label) #so now I have an array of detections
		end
		#turn that array of detections into an array of an array of detections sorted by receptive_field
		#temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
		#objects_observed[v, f] = temp_sorted_into_rfs
		objects_observed[v, f] = temp

		c = Camera_Params(camera_location = Coordinate(0.01,0.02,0.001), camera_focus = Coordinate(1.0,1.0,1.0))
		camera_trajectories[v, f] = c
	end
end



#count_observations(objects_observed)


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

gt_v = [0.5 0.0; 0.0 0.0; 0.5 0.0; 0.5 0.5; 0.5 0.1]
print(file, gt_v, " & ")
num_particles = 1

traces = unfold_particle_filter(num_particles, objects_observed, camera_trajectories, file)
println("done")
#visualize_observations(objects_observed, 1, 1, receptive_fields)
#visualize_trace(traces, 1, camera_trajectories, 1, 1, params)




# for v=1:4
# 	print_Vs_and_Rs_to_file(file, traces, num_particles, params, v)
# end
# print_Vs_and_Rs_to_file(file, traces, num_particles, params, 5, true)
#print_Vs_and_Rs_to_file(file, traces, num_particles, params, 1, true)
close(file)
