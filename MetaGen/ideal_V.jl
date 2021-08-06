#writes one CSV file for how an ideal observer would update the v matrix
#another CSV file for the ground truth V

using MetaGen
using JSON
using Pipe: @pipe

include("helper_function.jl")
include("scripts/useful_functions.jl")

#could equally use input dictionary
dict = @pipe "output.json" |> open |> read |> String |> JSON.parse

num_videos = 2
num_frames = 300

params = Video_Params(n_possible_objects = 8)
receptive_fields = make_receptive_fields()
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames)

###############################################################################
#Set up the output file
outfile = string("ideal_v_matrix.csv")
file = open(outfile, "w")

#return
print(file, "video_number&")
for i = 1:params.n_possible_objects
	print(file, "ideal_fa_", string(i), "&")
	print(file, "ideal_m_", string(i), "&")
end
print(file, "\n")

function print_helper(file, v::Int64, avg_v::Matrix{Float64})
	print(file, v, "&")
	n_rows,_ = size(avg_v)
    for i = 1:n_rows
        print(file, avg_v[i,1], "&")
        print(file, avg_v[i,2], "&")
    end
	print(file, "\n")
end

function print_ideal_v(file, dict::Array{Any}, params::Video_Params, camera_trajectories::Matrix{Camera_Params})
    num_videos, num_frames = size(camera_trajectories)

    #could make these zeros
    alphas = fill(1, (params.n_possible_objects,2))
    betas = fill(1, (params.n_possible_objects,2))
    avg_v = fill(0.0, (params.n_possible_objects,2))
    for v = 1:num_videos
        gt_objects = get_ground_truth(dict[v]["labels"]) #3D objects
        for f = 1:num_frames
            paramses = fill(params, length(gt_objects))
            camera_paramses = fill(camera_trajectories[v, f], length(gt_objects))
            gt_objects_2D = map(render, paramses, camera_paramses, gt_objects)
            gt_objects_2D = Array{Detection2D}(gt_objects_2D)

            inferences = convert(Vector{Int64}, dict[v]["views"][f]["inferences"]["labels"])
            println("inferences ", inferences)
            alphas, betas = update_alpha_beta(alphas, betas, inferences, gt_objects_2D)
        end
        println("alphas fa ", alphas[:,1])
        println("betas fa ", betas[:,1])
        avg_v[:,1] = alphas[:,1] ./ (betas[:,1].^2)#fa rate. mean of gamma distribution
        avg_v[:,2] = alphas[:,2] ./ (alphas[:,2] .+ betas[:,2])#miss rates. mean of beta distribution
        print_helper(file, v, avg_v)
    end
    return alphas, betas
end

alphas, betas = print_ideal_v(file, dict, params, camera_trajectories)
close(file)

################################################################################
#ground truth
file = open(string("ground_truth_V.csv"), "w")

print(file, "video_number&")
for i = 1:params.n_possible_objects
	print(file, "gt_fa_", string(i), "&")
	print(file, "gt_m_", string(i), "&")
end
print(file, "\n")

#for ground truth, would just subtract 1 from the alphas and betas matrices
gt_v = fill(0.0, (params.n_possible_objects,2))
gt_v[:,1] = (alphas[:,1] .- 1) ./ ((betas[:,1] .- 1).^2)
gt_v[:,2] = (alphas[:,2] .- 1) ./ ((alphas[:,2] .- 1) .+ (betas[:,2] .- 1))

for v=1:num_videos
	print_helper(file, v, avg_v)
end

close(file)
