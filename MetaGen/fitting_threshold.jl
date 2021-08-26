using JSON
using Pipe: @pipe
using MetaGen

include("helper_function.jl")
include("scripts/useful_functions.jl")

#dict = @pipe "../../metagen-data/data_labelled/data_labelled.json" |> open |> read |> String |> JSON.parse
dict = @pipe "../../scratch_work_07_16_21/08_26/data_labelled.json" |> open |> read |> String |> JSON.parse


num_videos = 10
num_frames = 200
ground_truth_world_states = get_ground_truth(dict, num_videos)

function fit_threshold_NN(num_videos::Int64, num_frames::Int64, dict::Any,
    ground_truth_world_states::Vector{Any}, threshold::Float64)

    params = Video_Params(n_possible_objects = 2)
    receptive_fields = make_receptive_fields(params)
    objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames, threshold)

    sim_NN = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
            #println("v ", v)
            #println("f ", f)
            gt_categories = threeD2twoD(ground_truth_world_states, v, f, params, camera_trajectories)
            NN_categories = last.(objects_observed[v,f])
            sim_NN[v] = sim_NN[v] + jaccard_similarity(gt_categories, NN_categories)
        end
        sim_NN[v] = sim_NN[v]/num_frames
    end
    return sum(sim_NN)/num_videos #averaging across videos
end

thresholds = collect(0:.01:1.0)
vs = fill(num_videos, length(thresholds))
fs = fill(num_frames, length(thresholds))
dicts = fill(dict, length(thresholds))
gts = fill(ground_truth_world_states, length(thresholds))
how_good = map(fit_threshold_NN, vs, fs, dicts, gts, thresholds)

println(thresholds[findmax(how_good)[2]])
