using JSON
using Pipe: @pipe
using MetaGen
using UnicodePlots

include("helper_function.jl")
include("scripts/useful_functions.jl")

#dict = @pipe "../../metagen-data/data_labelled/data_labelled.json" |> open |> read |> String |> JSON.parse
dict = @pipe "../../scratch_work_07_16_21/09_18/data_labelled.json" |> open |> read |> String |> JSON.parse


num_videos = 50
num_frames = 20
ground_truth_world_states = get_ground_truth(dict, num_videos)

function fit_threshold_NN(num_videos::Int64, num_frames::Int64, dict::Any,
    ground_truth_world_states::Vector{Any}, threshold::Float64)

    params = Video_Params(n_possible_objects = 5)
    receptive_fields = make_receptive_fields(params)
    objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames, threshold)

    sim_NN = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
            #println("v ", v)
            #println("f ", f)
            gt_categories = threeD2twoD_new(ground_truth_world_states[v], v, f, params, camera_trajectories)
            NN_categories = last.(objects_observed[v,f])
            #println(gt_categories)
            #println(NN_categories)
            sim_NN[v] = sim_NN[v] + jaccard_similarity(gt_categories, NN_categories)
            #println(jaccard_similarity(gt_categories, NN_categories)) #printing after doesn't work since running jaccard_similarity actually changes the inputs / outputs
        end
        sim_NN[v] = sim_NN[v]/num_frames
    end
    return sum(sim_NN)/num_videos #averaging across videos
end

thresholds = collect(0:.01:1.0)
#thresholds = collect([0.1])
vs = fill(num_videos, length(thresholds))
fs = fill(num_frames, length(thresholds))
dicts = fill(dict, length(thresholds))
gts = fill(ground_truth_world_states, length(thresholds))
how_good = map(fit_threshold_NN, vs, fs, dicts, gts, thresholds)

println(thresholds[findmax(how_good)[2]])

plt = lineplot(thresholds, how_good)
