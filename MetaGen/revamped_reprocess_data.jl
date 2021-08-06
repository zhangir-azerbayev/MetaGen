#script for pre-processing data. Ends with giving Jacard similarity between
#ground-truth world states and the mode inferred world state
using CSV
using DataFrames
using JSON
using Pipe: @pipe

include("helper_function.jl")

online_data = CSV.read("online_output.csv", DataFrame; delim = "&")
retro_data = CSV.read("retrospective_output.csv", DataFrame; delim = "&")
lesioned_data = CSV.read("lesioned_output.csv", DataFrame; delim = "&")

################################################################################
#could equally use output dictionary
dict = @pipe "0_data_detections.json" |> open |> read |> String |> JSON.parse

num_videos = 100

ground_truth_world_states = get_ground_truth(dict, num_videos)

################################################################################
online_world_states = new_parse_data(online_data, num_videos)
retrospective_world_states = new_parse_data(retro_data, num_videos)
lesioned_world_states = new_parse_data(lesioned_data, num_videos)

################################################################################
#get just object categories
ground_truth_categories = extract_category(ground_truth_world_states)
online_categories = extract_category(online_world_states)
retrospective_categories = extract_category(retrospective_world_states)
lesioned_categories = extract_category(lesioned_world_states)

#jaccard similarity
sim_online = map(jaccard_similarity, ground_truth_categories, online_categories)
sim_retrospective = map(jaccard_similarity, ground_truth_categories, retrospective_categories)
sim_lesioned = map(jaccard_similarity, ground_truth_categories, lesioned_categories)
################################################################################
#make new dataframe

new_df = DataFrame(video = 1:num_videos, sim_online = sim_online, sim_retrospective = sim_retrospective, sim_lesioned = sim_lesioned)
CSV.write("similarity.csv", new_df)
