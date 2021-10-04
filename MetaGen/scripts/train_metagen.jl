println("Starting")

using MetaGen
using JSON
import YAML
using Pipe: @pipe
using Random

config_path = ARGS[1]
config = YAML.load_file(config_path)
output_dir = "results" 
mkdir(output_dir)

include("useful_functions.jl")
dict = []
#for i = 0:config["batches_upto"]
	#to_add =  @pipe "$(config["input_file_dir"])$(i)_data_labelled.json" |> open |> read |> String |> JSON.parse
to_add =  @pipe "$(config["input_file"])" |> open |> read |> String |> JSON.parse
append!(dict, to_add)
#end
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse


#Random.seed!(15)
#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
num_videos = config["num_videos"]
num_frames = config["num_frames"]
threshold = config["threshold"]

categories_subset = config["categories_subset"]

params = Video_Params(n_possible_objects = length(categories_subset))
top_n = 5

receptive_fields = make_receptive_fields(params)
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, categories_subset, num_videos, num_frames, threshold, top_n)

num_particles = config["num_particles"]
mcmc_steps_outer = config["mcmc_steps_outer"]
mcmc_steps_inner = config["mcmc_steps_inner"]
shuffle_type = config["shuffle_type"]
################################################################################
#Online MetaGen

num_videos_train = convert(Int64, num_videos/2)
half = convert(Int64, num_videos_train/2)

if shuffle_type==0
	order = collect(1:num_videos_train)
elseif shuffle_type==1
	order = vcat(reverse(collect((half+1):num_videos_train)), reverse(collect(1:half)))
elseif shuffle_type==2
	order = vcat(collect((half+1):num_videos_train), collect(1:half))
elseif shuffle_type==3
	order = vcat(reverse(collect(1:half)), reverse(collect((half+1):num_videos_train)))
end

training_objects_observed = objects_observed[order, :]
training_camera_trajectories = camera_trajectories[order, :]

#Set up the output files
online_V_file = open(output_dir * "/online_V.csv", "w")
file_header_V(online_V_file, params)
online_ws_file = open(output_dir * "/online_ws.csv", "w")
file_header_ws(online_ws_file, params, num_particles)

println("start pf for online")

traces, inferred_world_states, avg_v = unfold_particle_filter(nothing,
	num_particles, mcmc_steps_outer, mcmc_steps_inner, training_objects_observed,
	training_camera_trajectories, params, online_V_file, online_ws_file, order)
close(online_V_file)
close(online_ws_file)

println("avg_v ", avg_v)
println("done with pf for online")

################################################################################
#Retrospective MetaGen

#training set and test set
order = vcat(order, (num_videos_train+1):num_videos)
input_objects_observed = vcat(objects_observed[order, :])
input_camera_trajectories = vcat(camera_trajectories[order, :])

#Set up the output file
retro_V_file = open(output_dir * "/retro_V.csv", "w")
file_header_V(retro_V_file, params)
retro_ws_file = open(output_dir * "/retro_ws.csv", "w")
file_header_ws(retro_ws_file, params, num_particles)

println("start retrospective")

traces, inferred_world_states, avg_v = unfold_particle_filter(avg_v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	input_objects_observed, input_camera_trajectories, params, retro_V_file, retro_ws_file, order)
close(retro_V_file)
close(retro_ws_file)

println("done with pf for retrospective")

#=

################################################################################
#run Lesioned MetaGen

#Set up the output file
lesioned_outfile = output_dir * "/lesioned_output.csv"
lesioned_file = open(lesioned_outfile, "w")
file_header(lesioned_file)

v = zeros(length(params.possible_objects), 2)
v[:,1] .= 1.0
v[:,2] .= 0.5
unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, lesioned_file)
close(lesioned_file)

println("done with pf for lesioned metagen")


=#

################################################################################
#for writing an output file for a demo using Retro MetaGen.

#undor re-ordering of inferred_world_states
inferred_world_states = inferred_world_states[order]

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_world_states, num_videos, num_frames)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open(output_dir * "/output.json","w") do f
	JSON.print(f,out)
end

println("finished writing json")
