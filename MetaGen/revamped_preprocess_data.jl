#script for pre-processing data. Ends with giving Jacard similarity between
#ground-truth world states and the mode inferred world state
using CSV
using DataFrames
using JSON
using Pipe: @pipe
using MetaGen

include("helper_function.jl")

online_data = CSV.read("../Data/online_output.csv", DataFrame; delim = "&")
#retro_data = CSV.read("../Data/retrospective_output.csv", DataFrame; delim = "&")
#lesioned_data = CSV.read("../Data/lesioned_output.csv", DataFrame; delim = "&")

################################################################################
#could equally use input or output dictionary
dict = @pipe "../Data/output.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/08_20/data_labelled.json" |> open |> read |> String |> JSON.parse


num_videos = 10

ground_truth_world_states = get_ground_truth(dict, num_videos)

################################################################################
online_world_states = new_parse_data(online_data, num_videos)
#retrospective_world_states = new_parse_data(retro_data, num_videos)
#lesioned_world_states = new_parse_data(lesioned_data, num_videos)

################################################################################
#get just object categories
ground_truth_categories = extract_category(ground_truth_world_states)
online_categories = extract_category(online_world_states)
#retrospective_categories = extract_category(retrospective_world_states)
#lesioned_categories = extract_category(lesioned_world_states)

#jaccard similarity
sim_online = map(jaccard_similarity, ground_truth_categories, online_categories)
#sim_retrospective = map(jaccard_similarity, ground_truth_categories, retrospective_categories)
#sim_lesioned = map(jaccard_similarity, ground_truth_categories, lesioned_categories)
################################################################################
#make new dataframe

#new_df = DataFrame(video = 1:num_videos, sim_online = sim_online, sim_retrospective = sim_retrospective, sim_lesioned = sim_lesioned)
new_df = DataFrame(video = 1:num_videos, sim_online = sim_online)
CSV.write("../Data/similarity3D.csv", new_df)

################################################################################
#repeat Jaccard sim in 2D
using MetaGen

include("scripts/useful_functions.jl")

num_videos = 10
num_frames = 200

function jacccard_sim_2D(num_videos::Int64, num_frames::Int64, dict::Any,
    ground_truth_world_states::Vector{Any}, online_world_states::Vector{Any})
    #retrospective_world_states::Vector{Any}, lesioned_world_states::Vector{Any})

    params = Video_Params(n_possible_objects = 7)
    receptive_fields = make_receptive_fields()
    objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames)

    sim_online = zeros(num_videos) #will contain jaccard sim for a video averaged across frames
    #sim_retrospective = zeros(num_videos)
    #sim_lesioned = zeros(num_videos)
    sim_NN = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
            println("v ", v)
            println("f ", f)
            gt_categories = threeD2twoD(ground_truth_world_states, v, f, params, camera_trajectories)

            online_categories = threeD2twoD(online_world_states, v, f, params, camera_trajectories)
            #retrospective_categories = threeD2twoD(retrospective_world_states, v, f, params, camera_trajectories)
            #lesioned_categories = threeD2twoD(lesioned_world_states, v, f, params, camera_trajectories)

            NN_categories = last.(objects_observed[v,f])

            sim_online[v] = sim_online[v] + jaccard_similarity(gt_categories, online_categories)
            #sim_retrospective[v] = sim_retrospective[v] + jaccard_similarity(gt_categories, retrospective_categories)
            #sim_lesioned[v] = sim_lesioned[v] + jaccard_similarity(gt_categories, lesioned_categories)
            sim_NN[v] = sim_NN[v] + jaccard_similarity(gt_categories, NN_categories)
        end
        sim_online[v] = sim_online[v]/num_frames
        #sim_retrospective[v] = sim_retrospective[v]/num_frames
        #sim_lesioned[v] = sim_lesioned[v]/num_frames
        sim_NN[v] = sim_NN[v]/num_frames
    end
    return sim_online, sim_NN#sim_retrospective, sim_lesioned, sim_NN
end

# sim_online, sim_retrospective, sim_lesioned, sim_NN = jacccard_sim_2D(num_videos,
#     num_frames, dict, ground_truth_world_states, online_world_states,
#     retrospective_world_states, lesioned_world_states)

sim_online, sim_NN = jacccard_sim_2D(num_videos,
    num_frames, dict, ground_truth_world_states, online_world_states)

new_df = DataFrame(video = 1:num_videos, sim_online = sim_online, sim_NN = sim_NN)
CSV.write("../Data/similarity2D.csv", new_df)
