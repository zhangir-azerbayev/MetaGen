#writes one CSV file for how an ideal observer would update the v matrix
#another CSV file for the ground truth V

using MetaGen
using JSON
using Pipe: @pipe

include("helper_function.jl")
include("scripts/useful_functions.jl")

path = "../../scratch_work_07_16_21/09_18/shuffle_0/"

#could equally use input dictionary
#dict = @pipe "output.json" |> open |> read |> String |> JSON.parse
dict = @pipe (path * "output.json") |> open |> read |> String |> JSON.parse


num_videos = 50
num_frames = 20
threshold = 0.0
top_n = 5

params = Video_Params(n_possible_objects = 5)
receptive_fields = make_receptive_fields(params)
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, num_videos, num_frames, threshold, top_n)

###############################################################################
#Set up the output file
outfile = string(path * "ideal_v_matrix.csv")
file = open(outfile, "w")

#return
print(file, "video_number&")
for i = 1:params.n_possible_objects
	print(file, "ideal_fa_", string(i), "&")
	print(file, "ideal_m_", string(i), "&")
end
print(file, "ground_truth_world_states")
print(file, "\n")

function print_helper(file, v::Int64, avg_v::Matrix{Float64}, gt_objects::Vector{Any})
	print(file, v, "&")
	n_rows,_ = size(avg_v)
    for i = 1:n_rows
        print(file, avg_v[i,1], "&")
        print(file, avg_v[i,2], "&")
    end
	print(file, gt_objects)
	print(file, "\n")
end

function print_ideal_v(file, dict::Array{Any}, params::Video_Params,
	camera_trajectories::Matrix{Camera_Params},
	objects_observed::Matrix{Array{Detection2D}})

    num_videos, num_frames = size(camera_trajectories)

	real_detections = Matrix{Array{Detection2D}}(undef, num_videos, num_frames)

    #could make these zeros
    alphas = fill(1, (params.n_possible_objects,2))
    betas = fill(1, (params.n_possible_objects,2))
    avg_v = fill(0.0, (params.n_possible_objects,2))

	gt_objects = get_ground_truth(dict, num_videos) #3D objects
    for v = 1:num_videos
		#println("v ", v)
		#println("gt_objects ", gt_objects[v])
		old_beta = deepcopy(betas)
		old_alpha = deepcopy(alphas)
        for f = 1:num_frames
            paramses = fill(params, length(gt_objects[v]))
            camera_paramses = fill(camera_trajectories[v, f], length(gt_objects[v]))
            gt_objects_2D = map(render, paramses, camera_paramses, gt_objects[v])
            gt_objects_2D = Array{Detection2D}(gt_objects_2D)

			#real_detections = filter!(within_frame, gt_objects_2D)


			#observations

            #inferences = convert(Vector{Int64}, dict[v]["views"][f]["inferences"]["labels"])
            #println("inferences ", inferences)
            #alphas, betas = update_alpha_beta(alphas, betas, inferences, gt_objects_2D)
			alphas, betas = update_alpha_beta(false, alphas, betas, objects_observed[v,f], gt_objects_2D)
        end
		change_alpha = alphas - old_alpha
		change_beta = betas - old_beta
		miss_rate = (change_alpha[:,2]) ./ ((change_alpha[:,2]) .+ (change_beta[:,2])) #don't need -1 since only looking at changes
		println(v)
		println(miss_rate)


        #println("alphas fa ", alphas[:,1])
        #println("betas fa ", betas[:,1])
		#println("alphas miss ", alphas[:,2])
        #println("betas miss ", betas[:,2])
        avg_v[:,1] = alphas[:,1] ./ (betas[:,1].^2)#fa rate. mean of gamma distribution
        avg_v[:,2] = alphas[:,2] ./ (alphas[:,2] .+ betas[:,2])#miss rates. mean of beta distribution
        print_helper(file, v, avg_v, gt_objects[v])
    end
    return alphas, betas
end


alphas, betas = print_ideal_v(file, dict, params, camera_trajectories, objects_observed)
close(file)

################################################################################
#ground truth
file = open(string(path * "ground_truth_V.csv"), "w")

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

gt_objects = get_ground_truth(dict, num_videos)
for v=1:num_videos
	print_helper(file, v, gt_v, gt_objects[v])
end

close(file)

################################################################################
#add ground truth per frame to dictionary
out = write_gt_to_dict(dict, camera_trajectories, gt_objects)

open(path * "output_with_gt.json","w") do f
	JSON.print(f,out)
end
