using MetaGen
using JSON
using Pipe: @pipe

include("useful_functions.jl")

dict = @pipe "../Data/data_first_10_detections.json" |> open |> read |> String |> JSON.parse

#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
params = Video_Params()

receptive_fields = make_receptive_fields()
objects_observed, camera_trajectories = make_observations(dict, receptive_fields)

num_particles = 100
traces = unfold_particle_filter(num_particles, objects_observed, camera_trajectories)



outfile = string("test.csv")
file = open(outfile, "w")
#file header
for v=1:9
	print(file, "avg V ", v, " & ")
    print(file, "dictionary realities PF for scene ", v, " & ")
	print(file, "mode realities PF for scene ", v, " & ")
end
print(file, "avg V 10 &")
print(file, "dictionary realities PF for scene 10 & ")
print(file, "mode realities PF for scene  10")

print(file, "\n")

for v=1:9
    print_Vs_and_Rs_to_file(file, traces, num_particles, params, v)
end
print_Vs_and_Rs_to_file(file, traces, num_particles, params, 10, true)
close(file)
