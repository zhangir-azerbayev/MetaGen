println("here")

using MetaGen
using JSON
import YAML
using Pipe: @pipe
using Random

config_path = ARGS[1]
config = YAML.load_file(config_path)
mkdir("results_marlene/$(config["experiment_name"])")

include("useful_functions.jl")
dict = []
for i = 0:config["batches_upto"]
	to_add =  @pipe "$(config["input_file_dir"])$(i)_data_labelled.json" |> open |> read |> String |> JSON.parse
	append!(dict, to_add)
end
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse


Random.seed!(15)
#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
num_videos = config["num_videos"]
num_frames = config["num_frames"]

params = Video_Params(n_possible_objects = 7)

receptive_fields = make_receptive_fields()
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames)

################################################################################
#Set up the output file
online_outfile = "results_marlene/$(config["experiment_name"])/online_output.csv"
online_file = open(online_outfile, "w")
file_header(online_file)

################################################################################
#Online MetaGen
num_particles = config["num_particles"]
mcmc_steps_outer = config["mcmc_steps_outer"]
mcmc_steps_inner = config["mcmc_steps_inner"]
#@profilehtml unfold_particle_filter(false, num_particles, objects_observed, camera_trajectories, params, file)
traces, inferred_realities, avg_v = unfold_particle_filter(nothing,
	num_particles, mcmc_steps_outer, mcmc_steps_inner, objects_observed,
	camera_trajectories, params, file)
close(online_file)

println("done with pf for online & retrospective metagen")

################################################################################
#Retrospective MetaGen

#Set up the output file
retro_outfile = "results_marlene/$(config["experiment_name"])/retrospective_output.csv"
retro_file = open(retro_outfile, "w")
file_header(retro_file)

unfold_particle_filter(avg_v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, file)
close(retro_file)

################################################################################
#run Lesioned MetaGen

#Set up the output file
lesioned_outfile = "results_marlene/$(config["experiment_name"])/lesioned_output.csv"
lesioned_file = open(lesioned_outfile, "w")
file_header(lesioned_file)

v = zeros(length(params.possible_objects), 2)
v[:,1] .= 1.0
v[:,2] .= 0.5
unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, file)
close(lesioned_file)

println("done with pf for lesioned metagen")

################################################################################
#for writing an output file for a demo using Retrospective MetaGen

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_realities, num_videos, num_frames)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open("results_marlene/$(config["experiment_name"])/output.json","w") do f
	JSON.print(f,out)
end

println("finished writing json")
