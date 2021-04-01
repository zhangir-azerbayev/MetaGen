using MetaGen
using JSON
using Pipe: @pipe

include("useful_functions.jl")

dict = @pipe "../Data/data_first_10_detections.json" |> open |> read |> String |> JSON.parse

#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
permanent_camera_params = Permanent_Camera_Params()

receptive_fields = make_receptive_fields(permanent_camera_params)
objects_observed, camera_trajectories = make_observations(dict, receptive_fields)

#num_particles = 5
#tr = unfold_particle_filter(num_particles, objects_observed, camera_trajectories);
