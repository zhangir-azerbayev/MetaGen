#helper functions for pre-processing data
using DataFrames

COCO_CLASSES = ["person", "bicycle", "car", "motorcycle",
			"airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
			"N/A", "stop sign","parking meter", "bench", "bird", "cat", "dog", "horse",
			"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
			"umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
			"sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
			"surfboard", "tennis racket","bottle", "N/A", "wine glass", "cup", "fork", "knife",
			"spoon","bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
			"hot dog", "pizza","donut", "cake", "chair", "couch", "potted plant", "bed",
			"N/A", "dining table","N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
			"remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
			"book","clock", "vase", "scissors", "teddy bear", "hair drier",
			"toothbrush"]

#for now, remove books
#office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]
office_subset = ["chair", "microwave"]

# dictionary_scenenet_to_office = Dict("chair" => "chair",
# "straightchair" => "chair", "swivelchair" => "chair", "windsorchair" => "chair",
# "cantileverchair" => "chair", "lawnchair" => "chair", "armchair" => "chair",
# "armchair" => "chair",
# "keyboard" => "keyboard", "computerkeyboard" => "keyboard",
# "laptop" => "laptop", "draftingtable" => "dining table",
# "table" => "dining table",
# "plant" => "potted plant",
# "cellphone" => "cell phone", "cellulartelephone" => "cell phone",
# "bottle" => "bottle", "winebottle" => "bottle")

#dictionary_gt_to_office = Dict("chair" => "chair", "sofa" => "couch")
dictionary_gt_to_office = Dict("chair" => "chair", "microwave" => "microwave")



#helper function for getting ground-truth from dictionary
function get_ground_truth(dict::Array{Any}, num_videos::Int64)
    world_states = Array{Any}(undef, num_videos)
    for v = 1:num_videos
		#println(v)
        world_state = []
        for item = 1:length(dict[v]["labels"])
			scene_net_name = dict[v]["labels"][item]["category_name"]
			#println(scene_net_name)
			office_name = get(dictionary_gt_to_office, scene_net_name, "NA") #office name will be NA if it's not an entry in the dictionary
			#println(office_name)
			label = findfirst(office_subset .== office_name)
            location = dict[v]["labels"][item]["position"]
			#println(location)
            if !isnothing(label)
                push!(world_state, (location[1], location[2], location[3], label))
            end
        end
        world_states[v] = world_state
    end
    return world_states
end

################################################################################
function extract_category(world_states::Any)
    categories_only = deepcopy(world_states)
    for v = 1:num_videos
        for i = 1:length(world_states[v])
            categories_only[v][i] = world_states[v][i][4] #get just the category
        end
    end
    return categories_only
end

################################################################################
function threeD2twoD(threeDobjects::Vector{Any}, v::Int64, f::Int64,
		params::Video_Params, camera_trajectories::Matrix{Camera_Params})
	paramses = fill(params, length(threeDobjects[v]))
	camera_paramses = fill(camera_trajectories[v, f], length(threeDobjects[v]))
	threeDobjects_2D = map(render, paramses, camera_paramses, threeDobjects[v])
	threeDobjects_2D = Array{Detection2D}(threeDobjects_2D)
	threeDobjects_2D = filter(p -> within_frame(p), threeDobjects_2D)
	categories = last.(threeDobjects_2D)
end

################################################################################
#given a string, parse the columns of the dataframe with that string and return
#a vector of world states
function parse_data(data::DataFrame, model_name::String, num_videos::Int64)
    names_list = []
    for i = 1:length(names(data))
        if occursin(model_name, names(data)[i])
            push!(names_list, names(data)[i])
        end
    end

    world_states = Array{Any}(undef, length(names_list)) #length(online_names) should be the same as num_videos
    for i = 1:length(names_list)
        world_states[i] = eval(Meta.parse(data[1, Symbol(names_list[i])])) #1 is for the first row. we only have 1 row
    end
    return world_states
end

################################################################################
#parse the dataframe and return a vector of world states
function new_parse_data(data::DataFrame, num_videos::Int64)
    world_states = Array{Any}(undef, num_videos) #length(online_names) should be the same as num_videos
    for i = 1:num_videos
        world_states[i] = eval(Meta.parse(data[i, "inferred_mode_realities"])) #1 is for the first row. we only have 1 row
    end
    return world_states
end

################################################################################
#Jaccard similarity
function jaccard_similarity(a::Union{Array{Any}, Vector{Int64}}, b::Union{Array{Any}, Vector{Int64}})
    overlap_of_sets = []
    union_of_sets = []
    for i = 1:length(a)
        index = findfirst(b .== a[i])
        if !isnothing(index)
            #remove from b, add to overlap
            entry = splice!(b, index)
            push!(overlap_of_sets, entry)
        end
        #add a value to union
        push!(union_of_sets, a[i])
    end

    #add whatever is left of b
    for j = 1:length(b)
        push!(union_of_sets, b[j])
    end

    #what happens when both a and b were empty?
    if length(union_of_sets) == 0
        return 1
    else
        return length(overlap_of_sets)/length(union_of_sets)
    end
end

################################################################################

function within_frame(x::Float64, y::Float64)
    x >= 0 && x <= 256 && y >= 0 && y <= 256 #hard-codded frame size
end

function within_frame(p::Detection2D)
    p[1] >= 0 && p[1] <= 256 && p[2] >= 0 && p[2] <= 256 #hard-codded frame size
end

################################################################################
#for making ground-truth demo
function write_gt_to_dict(dict::Array{Any,1}, camera_trajectories::Matrix{Camera_Params}, gt_objects::Vector{Any})
    params = Video_Params()
	num_videos, num_frames = size(camera_trajectories)

    #add inferences about the objects in the scenes
    for v = 1:num_videos
		#println("v ", v)
        #for each object
        gt = gt_objects[v]

        #add per frame
        for f = 1:num_frames
			#println("f ", f)
            labels = []
            centers = []
            for i in 1:length(gt)
                r = gt[i]
                x, y = get_image_xy(camera_trajectories[v,f], params, Coordinate(r[1],r[2],r[3]))
                if within_frame(x, y)
					#println(r)
                    push!(labels, r[4])
                    push!(centers, (x, y))
                end
            end
            d_f = dict[v]["views"][f]
            a = Dict("labels" => labels, "centers" => centers)
            merge!(d_f, Dict("ground_truth" => a))
        end
    end

    return dict
end
