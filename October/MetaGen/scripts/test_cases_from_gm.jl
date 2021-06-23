#for running test cases from the generative model
using MetaGen
using Gen

################################################################################
num_videos = 1
num_frames = 2
gt_trace,_ = Gen.generate(metacog, (num_videos, num_frames))
#println(gt_trace)
gt_choices = get_choices(gt_trace)

receptive_fields = make_receptive_fields()
num_receptive_fields = length(receptive_fields)

objects_observed = Matrix{Array{Array{Detection2D}}}(undef, num_videos, num_frames)
camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)
for v= 1:num_videos
	for f = 1:num_frames
		temp = []
		for rec_field = 1:num_receptive_fields
			println(gt_choices[:videos => v => :frame_chain => f => rec_field => :observations_2D])
			vcat(temp, convert(Array{Detection2D}, gt_choices[:videos => v => :frame_chain => f => rec_field => :observations_2D]))
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

visualize_observations(objects_observed, 1, 1, receptive_fields)
visualize_observations(objects_observed, 1, 2, receptive_fields)

num_particles = 100
traces = unfold_particle_filter(num_particles, objects_observed, camera_trajectories, num_receptive_fields)
println("done")
#visualize_observations(objects_observed, 1, 1, receptive_fields)
#visualize_trace(traces, 1, camera_trajectories, 1, 1, params)

outfile = string("test_case.csv")
file = open(outfile, "w")
#file header
# for v=1:1
# 	print(file, "avg V ", v, " & ")
#     print(file, "dictionary realities PF for scene ", v, " & ")
# 	print(file, "mode realities PF for scene ", v, " & ")
# end
print(file, "avg V 1 &")
print(file, "dictionary realities PF for scene 1 & ")
print(file, "mode realities PF for scene  1")

print(file, "\n")

# for v=1:1
#     print_Vs_and_Rs_to_file(file, traces, num_particles, params, v)
# end
# print_Vs_and_Rs_to_file(file, traces, num_particles, params, 2, true)
print_Vs_and_Rs_to_file(file, traces, num_particles, params, 1, true)
close(file)
