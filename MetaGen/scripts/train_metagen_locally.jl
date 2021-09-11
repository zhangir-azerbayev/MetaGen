println("here")

using MetaGen
using JSON
import YAML
using Pipe: @pipe
using Random

# config_path = ARGS[1]
# config = YAML.load_file(config_path)
# mkdir("results_marlene/$(config["experiment_name"])")

include("useful_functions.jl")
#dict = []
# for i = 0:config["batches_upto"]
# 	to_add =  @pipe "$(config["input_file_dir"])$(i)_data_labelled.json" |> open |> read |> String |> JSON.parse
# 	append!(dict, to_add)
# end
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/metagen-data/data_labelled/data_labelled.json" |> open |> read |> String |> JSON.parse
path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/09_06/"

dict = @pipe (path * "data_labelled.json") |> open |> read |> String |> JSON.parse

#Random.seed!(17) #15 produces -Inf for a particle from video 1, frame 111, no rejuvination steps
#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
num_videos = 2
num_frames = 200
threshold = 0.24

num_particles = 100
mcmc_steps_outer = 1
mcmc_steps_inner = 1

params = Video_Params(n_possible_objects = 3)

receptive_fields = make_receptive_fields(params)
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames, threshold)

################################################################################

#Set up the output files
online_V_file = open(path * "online_V.csv", "w")
file_header_V(online_V_file, params)
online_ws_file = open(path * "online_ws.csv", "w")
file_header_ws(online_ws_file, params, num_particles)


################################################################################
#Online MetaGen
#@profilehtml unfold_particle_filter(false, num_particles, objects_observed, camera_trajectories, params, file)

traces, inferred_world_states, avg_v = unfold_particle_filter(nothing,
	num_particles, mcmc_steps_outer, mcmc_steps_inner, objects_observed,
	camera_trajectories, params, online_V_file, online_ws_file)
close(online_V_file)
close(online_ws_file)
#
println("done with pf for online")

################################################################################



# #Retrospective MetaGen
#
#Set up the output file
retro_V_file = open(path * "retro_V.csv", "w")
file_header_V(retro_V_file, params)
retro_ws_file = open(path * "retro_ws.csv", "w")
file_header_ws(retro_ws_file, params, num_particles)

unfold_particle_filter(avg_v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, retro_V_file, retro_ws_file)
close(retro_V_file)
close(retro_ws_file)


# ################################################################################
# #Lesioned MetaGen
#
#Set up the output file
lesioned_V_file = open(path * "lesioned_V.csv", "w")
file_header_V(lesioned_V_file, params)
lesioned_ws_file = open(path * "lesioned_ws.csv", "w")
file_header_ws(lesioned_ws_file, params, num_particles)

v = zeros(length(params.possible_objects), 2)
v[:,1] .= 1.0
v[:,2] .= 0.5
unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, lesioned_V_file, lesioned_ws_file)
close(lesioned_V_file)
close(lesioned_ws_file)

println("done with pf for lesioned metagen")

################################################################################
#for writing an output file for a demo using Online MetaGen

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_world_states, num_videos, num_frames)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open(path * "output.json","w") do f
	JSON.print(f,out)
end

println("finished writing json")
