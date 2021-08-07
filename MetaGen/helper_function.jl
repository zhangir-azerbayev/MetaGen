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
office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]

dictionary_scenenet_to_office = Dict("chair" => "chair",
"straightchair" => "chair", "swivelchair" => "chair", "windsorchair" => "chair",
"cantileverchair" => "chair", "keyboard" => "keyboard", "computerkeyboard" => "keyboard",
"laptop" => "laptop", "table" => "dining table", "plant" => "potted plant",
"cellphone" => "cell phone", "cellulartelephone" => "cell phone",
"bottle" => "bottle", "winebottle" => "bottle")



#helper function for getting ground-truth from dictionary
function get_ground_truth(dict::Array{Any}, num_videos::Int64)
    world_states = Array{Any}(undef, num_videos)
    for v = 1:num_videos
        world_state = []
        for item = 1:length(dict[v]["labels"])
			scene_net_name = dict[v]["labels"][item]["category_name"]
			println(scene_net_name)
			office_name = get(dictionary_scenenet_to_office, scene_net_name, "NA") #office name will be NA if it's not an entry in the dictionary
            label = findfirst(office_subset .== office_name)
            location = dict[v]["labels"][item]["position"]
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
    categories_only = world_states
    for v = 1:num_videos
        for i = 1:length(world_states[v])
            categories_only[v][i] = world_states[v][i][4] #get just the category
        end
    end
    return categories_only
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
function jaccard_similarity(a::Array{Any}, b::Array{Any})
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
