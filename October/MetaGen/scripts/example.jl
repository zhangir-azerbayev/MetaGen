using MetaGen
using Gen

#possible_objects = ["person","bicycle","car","motorcycle","airplane"]
possible_objects = [1, 2, 3, 4, 5]
#call it


#include("useful_functions.jl")

dict = @pipe "../Data/data_first_10_detections.json" |> open |> read |> String |> JSON.parse

#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
receptive_fields = make_receptive_fields()
objects_observed, camera_trajectories = make_observations(dict, receptive_fields)

#constrain
obs = Gen.choicemap()
for v=1:10
    for f=1:6
        obs[:videos => v => :frame_chain => f => :camera => :camera_location_x] = camera_trajectories[v,f].camera_location.x
    end
end

gt_trace,log_score = Gen.generate(metacog, (possible_objects, 10, 6), obs)
gt_choices = get_choices(gt_trace)

#@show gt_choices
@show log_score
