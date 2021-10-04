println("here")

using MetaGen
using JSON
import YAML
using Pipe: @pipe
using Random

include("useful_functions.jl")

path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/09_18/"

dict = @pipe (path * "data_labelled.json") |> open |> read |> String |> JSON.parse

################################################################################
#Set up the observations
num_videos = 4 #total, including test set
num_frames = 20
threshold = 0.05

n_top = 5
params = Video_Params(n_possible_objects = 5)

receptive_fields = make_receptive_fields(params)
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames, threshold, n_top)

################################################################################
#Set up for MCMC
num_particles = 10
mcmc_steps_outer = 1
mcmc_steps_inner = 1


################################################################################
#Online MetaGen
num_videos_train = convert(Int64, num_videos/2)
@assert num_videos_train==50

if shuffle_type==0
	order = collect(1:num_videos_train)
elseif shuffle_type==1
	order = vcat(reverse(collect(26:50)), reverse(collect(1:25)))
elseif shuffle_type==2
	order = vcat(collect(26:50), collect(1:25))
elseif shuffle_type==3
	order = vcat(reverse(collect(1:25)), reverse(collect(26:50)))
end

training_objects_observed = objects_observed[order, :]
training_camera_trajectories = camera_trajectories[order, :]

#Set up the output files
online_V_file = open(path * "online_V.csv", "w")
file_header_V(online_V_file, params)
online_ws_file = open(path * "online_ws.csv", "w")
file_header_ws(online_ws_file, params, num_particles)

traces, inferred_world_states, avg_v = unfold_particle_filter(nothing,
	num_particles, mcmc_steps_outer, mcmc_steps_inner, training_objects_observed,
	training_camera_trajectories, params, online_V_file, online_ws_file, order)
close(online_V_file)
close(online_ws_file)
#
println("avg_v ", avg_v)
println("done with pf for online")

################################################################################

# #Retrospective MetaGen
#
#Set up the output file
retro_V_file = open(path * "retro_V.csv", "w")
file_header_V(retro_V_file, params)
retro_ws_file = open(path * "retro_ws.csv", "w")
file_header_ws(retro_ws_file, params, num_particles)

#training set and test set
order = vcat(order, 51:100)
input_objects_observed = vcat(objects_observed[order, :])
input_camera_trajectories = vcat(camera_trajectories[order, :])

traces, inferred_world_states, avg_v = unfold_particle_filter(avg_v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	input_objects_observed, input_camera_trajectories, params, retro_V_file, retro_ws_file, order)
close(retro_V_file)
close(retro_ws_file)


# # ################################################################################
# # #Lesioned MetaGen
# #
# #Set up the output file
# lesioned_V_file = open(path * "lesioned_V.csv", "w")
# file_header_V(lesioned_V_file, params)
# lesioned_ws_file = open(path * "lesioned_ws.csv", "w")
# file_header_ws(lesioned_ws_file, params, num_particles)
#
# v = zeros(length(params.possible_objects), 2)
# v[:,1] .= 1.0
# v[:,2] .= 0.5
# unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
# 	objects_observed, camera_trajectories, params, lesioned_V_file, lesioned_ws_file)
# close(lesioned_V_file)
# close(lesioned_ws_file)
#
# println("done with pf for lesioned metagen")

################################################################################
#for writing an output file for a demo using Retro MetaGen. will only make sense for non-mixed up version

#undor re-ordering of inferred_world_states
inferred_world_states = inferred_world_states[order]

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_world_states, num_videos, num_frames)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open(path * "output.json","w") do f
	JSON.print(f,out)
end

println("finished writing json")
