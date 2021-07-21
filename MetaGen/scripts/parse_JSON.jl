using MetaGen
using JSON
using Pipe: @pipe
using Random

include("useful_functions.jl")

dict = @pipe "../metagen_data/labelled_data/0/tiny_set_detections.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/tiny_set_detections.json" |> open |> read |> String |> JSON.parse

Random.seed!(15)
#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
num_videos = 2
num_frames = 300

params = Video_Params()

receptive_fields = make_receptive_fields()
objects_observed, camera_trajectories = make_observations(dict, receptive_fields)

#count_observations(objects_observed)

outfile = string("test.csv")
file = open(outfile, "w")

num_particles = 1
traces = unfold_particle_filter(num_particles, objects_observed, camera_trajectories, file)

println("done")


#file header
for v=1:1
	print(file, "avg V ", v, " & ")
    print(file, "dictionary realities PF for scene ", v, " & ")
	print(file, "mode realities PF for scene ", v, " & ")
end
print(file, "avg V 2 &")
print(file, "dictionary realities PF for scene 2 & ")
print(file, "mode realities PF for scene  2")

print(file, "\n")

inferred_realities = Array{Any}(undef, num_videos)
for v=1:1
    inferred_realities[v] = print_Vs_and_Rs_to_file(file, traces, num_particles, params, v)
end
inferred_realities[2] = print_Vs_and_Rs_to_file(file, traces, num_particles, params, 2, true)
close(file)

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_realities)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open("output_tiny_set_detections.json","w") do f
	JSON.print(f,out)
end
