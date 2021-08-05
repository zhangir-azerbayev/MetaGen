println("here")

using MetaGen
using JSON
import YAML
using Pipe: @pipe
using Random

config_path = ARGS[1]
config = YAML.load_file(config_path)

include("useful_functions.jl")

dict = @pipe config["input_file"] |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse


Random.seed!(15)
#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
num_videos = config["num_videos"]
num_frames = config["num_frames"]

params = Video_Params(n_possible_objects = 8)

receptive_fields = make_receptive_fields()
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames)

################################################################################
#Set up the output file
outfile = config["output_csv"]
file = open(outfile, "w")

#file header
for v= 1:num_videos
	print(file, "online avg V ", v, " & ")
    print(file, "online dictionary realities PF for scene ", v, " & ")
	print(file, "online mode realities PF for scene ", v, " & ")
end

for v= 1:num_videos
	print(file, "retrospective avg V ", v, " & ")
    print(file, "retrospective dictionary realities PF for scene ", v, " & ")
	print(file, "retrospective mode realities PF for scene ", v, " & ")
end

for v= 1:(num_videos-1)
	print(file, "lesioned avg V ", v, " & ")
    print(file, "lesioned dictionary realities PF for scene ", v, " & ")
	print(file, "lesioned mode realities PF for scene ", v, " & ")
end
print(file, "lesioned avg V ", num_videos, " & ")
print(file, "lesioned dictionary realities PF for scene ", num_videos, " & ")
print(file, "lesioned mode realities PF for scene ", num_videos)

print(file, "\n")

################################################################################
#Online MetaGen
num_particles = config["num_particles"]
mcmc_steps_outer = config["mcmc_steps_outer"]
mcmc_steps_inner = config["mcmc_steps_inner"]
#@profilehtml unfold_particle_filter(false, num_particles, objects_observed, camera_trajectories, params, file)
traces, inferred_realities, avg_v = unfold_particle_filter(nothing,
	num_particles, mcmc_steps_outer, mcmc_steps_inner, objects_observed,
	camera_trajectories, params, file)

println("done with pf for online & retrospective metagen")

################################################################################
#Retrospective MetaGen
unfold_particle_filter(avg_v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, file)


################################################################################
#run Lesioned MetaGen
v = zeros(length(params.possible_objects), 2)
v[:,1] .= 1.0
v[:,2] .= 0.5
unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, file)


println("done with pf for lesioned metagen")

close(file)

################################################################################
#for writing an output file for a demo using Retrospective MetaGen

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_realities, num_videos, num_frames)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open(config["output_json"],"w") do f
	JSON.print(f,out)
end

println("finished writing json")
