function file_header_V(file, params::Video_Params)
    #set up file header
    print(file, "order_run&")
    print(file, "video_number&")
    for i = 1:params.n_possible_objects
    	print(file, "fa_", string(i), "&")
    	print(file, "m_", string(i), "&")
    end

    for j = 1:num_particles
        for i = 1:params.n_possible_objects
        	print(file, "fa_", string(i), "_", string(j), "&")
        	print(file, "m_", string(i), "_", string(j), "&")
        end
        print(file, "weight_", string(j), "&")
    end
    print(file, "inferred_best_world_state")
    print(file, "\n")
end

function file_header_ws(file, params::Video_Params, num_particles::Int64)
    #set up file header
    print(file, "order_run&")
    print(file, "video_number&")
    print(file, "inferred_best_world_state&")
    for i = 1:num_particles-1
        print(file, "world_state_", string(i), "&")
        print(file, "weight_", string(i), "&")
    end
    print(file, "world_state_", num_particles, "&")
    print(file, "weight_", num_particles)
    print(file, "\n")
end



function file_header(file, params::Video_Params, num_particles::Int64)
    #set up file header
    print(file, "video_number&")
    for i = 1:params.n_possible_objects
    	print(file, "fa_", string(i), "&")
    	print(file, "m_", string(i), "&")
    end
    print(file, "inferred_world_states&inferred_best_world_state&")
    for i = 1:num_particles-1
        print(file, "world_state_", string(i), "&")
        print(file, "weight_", string(i), "&")
    end
    print(file, "world_state_", num_particles, "&")
    print(file, "weight_", num_particles)
    print(file, "\n")
end


function make_observations_full_COCO(dict::Array{Any,1}, receptive_fields::Vector{Receptive_Field}, num_videos::Int64, num_frames::Int64)
    objects_observed = Matrix{Array{Detection2D}}(undef, num_videos, num_frames)
    #getting undefined reference when I change to Array{Array{}} instead of matrix

    camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)

    #temporary for getting min and max values of positions
    x_min = 0
    x_max = 0
    y_min = 3
    y_max = 0
    z_min = 0
    z_max = 0

    f_x_min = 0
    f_x_max = 0
    f_y_min = 3
    f_y_max = 0
    f_z_min = 0
    f_z_max = 0

    labels_max = 0

    for v=1:num_videos
        for f=1:num_frames
            #indices is where confidence was > 0.5
            indices = dict[v]["views"][f]["detections"]["scores"] .> 0.5
            labels = round.(Int64, dict[v]["views"][f]["detections"]["labels"][indices])
            center = dict[v]["views"][f]["detections"]["center"][indices]
            temp = Array{Detection2D}(undef, length(labels))
            if length(labels) > 5
                println("uh oh. too many obs ", length(labels))
                println("video ", v)
                println("frame ", f)
            end
            for i = 1:length(labels)
                label = labels[i]
                x = center[i][1]
                y = center[i][2]
                if x < 0 || x > 320
                    println("uh oh. x ", x)
                    println("video ", v)
                    println("frame ", f)
                elseif y < 0 || y > 240
                    println("uh oh. y ", y)
                    println("video ", v)
                    println("frame ", f)
                end
                temp[i] = (x, y, label) #so now I have an array of detections
            end
            #turn that array of detections into an array of an array of detections sorted by receptive_field
            #temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
            objects_observed[v, f] = temp

            #camera trajectory
            x = dict[v]["views"][f]["camera"][1]
            y = dict[v]["views"][f]["camera"][2]
            z = dict[v]["views"][f]["camera"][3]
            #focus
            f_x = dict[v]["views"][f]["lookat"][1]
            f_y = dict[v]["views"][f]["lookat"][2]
            f_z = dict[v]["views"][f]["lookat"][3]
            c = Camera_Params(camera_location = Coordinate(x,y,z), camera_focus = Coordinate(f_x,f_y,f_z))
            camera_trajectories[v, f] = c

            length(labels) > labels_max ? labels_max = length(labels) : labels_max = labels_max

            x > x_max ? x_max = x : x_max = x_max
            x < x_min ? x_min = x : x_min = x_min
            y > y_max ? y_max = y : y_max = y_max
            y < y_min ? y_min = y : y_min = y_min
            z > z_max ? z_max = z : z_max = z_max
            z < z_min ? z_min = z : z_min = z_min

            f_x > f_x_max ? f_x_max = f_x : f_x_max = f_x_max
            f_x < f_x_min ? f_x_min = f_x : f_x_min = f_x_min
            f_y > f_y_max ? f_y_max = f_y : f_y_max = f_y_max
            f_y < f_y_min ? f_y_min = f_y : f_y_min = f_y_min
            f_z > f_z_max ? f_z_max = f_z : f_z_max = f_z_max
            f_z < f_z_min ? f_z_min = f_z : f_z_min = f_z_min

        end
    end

    println("labels_max ", labels_max)

    println("x_max ", x_max)
    println("x_min ", x_min)
    println("y_max ", y_max)
    println("y_min ", y_min)
    println("z_max ", z_max)
    println("z_min ", z_min)

    println("f_x_max ", f_x_max)
    println("f_x_min ", f_x_min)
    println("f_y_max ", f_y_max)
    println("f_y_min ", f_y_min)
    println("f_z_max ", f_z_max)
    println("f_z_min ", f_z_min)

    return objects_observed, camera_trajectories
end

function make_observations_office(dict::Array{Any,1}, receptive_fields::Vector{Receptive_Field}, office_subset::Vector{String}, num_videos::Int64, num_frames::Int64, threshold=0.5, top_n=Inf)

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

    #office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]
    #office_subset = ["book", "chair", "keyboard", "laptop", "table", "potted plant", "cell phone", "wine bottle"]
    #office_subset = ["dining table", "microwave", "backpack", "bowl", "couch"]

    objects_observed = Matrix{Array{Detection2D}}(undef, num_videos, num_frames)
    #getting undefined reference when I change to Array{Array{}} instead of matrix

    camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)

    #temporary for getting min and max values of positions
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    z_min = 0
    z_max = 0

    f_x_min = 0
    f_x_max = 0
    f_y_min = 3
    f_y_max = 0
    f_z_min = 0
    f_z_max = 0

    labels_max = 0

    for v=1:num_videos
        for f=1:num_frames
            #first sort by confidence level
            idx = sortperm(-1 .* dict[v]["views"][f]["detections"]["scores"]) #descending order
            dict[v]["views"][f]["detections"]["scores"] = dict[v]["views"][f]["detections"]["scores"][idx]
            dict[v]["views"][f]["detections"]["center"] = dict[v]["views"][f]["detections"]["center"][idx]
            dict[v]["views"][f]["detections"]["labels"] = dict[v]["views"][f]["detections"]["labels"][idx]

            indices = dict[v]["views"][f]["detections"]["scores"] .> threshold #indices should be sorted by confidence
            arr = round.(Int64, dict[v]["views"][f]["detections"]["labels"][indices]) #for turning to ints
            center = dict[v]["views"][f]["detections"]["center"][indices]
            #temp = Array{Detection2D}(undef, length(labels))
            # if length(labels) > 5
            #     println("uh oh. too many obs ", length(labels))
            #     println("video ", v)
            #     println("frame ", f)
            # end
            temp = []
            for i = 1:length(arr)
                label = findfirst(COCO_CLASSES[arr[i]] .== office_subset)
                if !isnothing(label)
                    x = center[i][1]
                    y = center[i][2]
                    push!(temp, (x, y, label))
                end
            end

            if length(temp) > 5
                println("v ", v, " f ", f)
                println(length(temp))
            end

            if length(temp) > top_n #temp should already be sorted by confidence. if it's not, will have to add code to sort it first
                #println(length(temp))
                #println("v ", v, " f ", f)
                println("v gets trimmed ", v)
                temp = temp[1:top_n]
            end

            #turn that array of detections into an array of an array of detections sorted by receptive_field
            #temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
            objects_observed[v, f] = convert(Array{Detection2D}, temp)

            #camera trajectory
            # x = dict[v]["views"][f]["camera"][1]
            # y = dict[v]["views"][f]["camera"][2]
            # z = dict[v]["views"][f]["camera"][3]
            x = dict[v]["views"][f]["camera"]["x"]
            y = dict[v]["views"][f]["camera"]["y"]
            z = dict[v]["views"][f]["camera"]["z"]
            #focus
            #f_x = dict[v]["views"][f]["lookat"][1]
            #f_y = dict[v]["views"][f]["lookat"][2]
            #f_z = dict[v]["views"][f]["lookat"][3]
            f_x = dict[v]["views"][f]["lookat"]["x"]
            f_y = dict[v]["views"][f]["lookat"]["y"]
            f_z = dict[v]["views"][f]["lookat"]["z"]
            c = Camera_Params(camera_location = Coordinate(x,y,z), camera_focus = Coordinate(f_x,f_y,f_z))
            camera_trajectories[v, f] = c

            length(temp) > labels_max ? labels_max = length(temp) : labels_max = labels_max

            x > x_max ? x_max = x : x_max = x_max
            x < x_min ? x_min = x : x_min = x_min
            y > y_max ? y_max = y : y_max = y_max
            y < y_min ? y_min = y : y_min = y_min
            z > z_max ? z_max = z : z_max = z_max
            z < z_min ? z_min = z : z_min = z_min

            f_x > f_x_max ? f_x_max = f_x : f_x_max = f_x_max
            f_x < f_x_min ? f_x_min = f_x : f_x_min = f_x_min
            f_y > f_y_max ? f_y_max = f_y : f_y_max = f_y_max
            f_y < f_y_min ? f_y_min = f_y : f_y_min = f_y_min
            f_z > f_z_max ? f_z_max = f_z : f_z_max = f_z_max
            f_z < f_z_min ? f_z_min = f_z : f_z_min = f_z_min

        end
    end

    println("labels_max ", labels_max)
    #
    println("x_max ", x_max)
    println("x_min ", x_min)
    println("y_max ", y_max)
    println("y_min ", y_min)
    println("z_max ", z_max)
    println("z_min ", z_min)

    println("f_x_max ", f_x_max)
    println("f_x_min ", f_x_min)
    println("f_y_max ", f_y_max)
    println("f_y_min ", f_y_min)
    println("f_z_max ", f_z_max)
    println("f_z_min ", f_z_min)

    return objects_observed, camera_trajectories
end

#for testing: feed ground-truth 2D detections
function make_observations_office_from_gt(dict::Array{Any,1}, receptive_fields::Vector{Receptive_Field}, office_subset::Vector{String},
    num_videos::Int64, num_frames::Int64, threshold=0.5)

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

    #office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]
    #office_subset = ["book", "chair", "keyboard", "laptop", "table", "potted plant", "cell phone", "wine bottle"]
    #office_subset = ["dining table", "microwave", "backpack", "bowl", "couch"]

    objects_observed = Matrix{Array{Detection2D}}(undef, num_videos, num_frames)
    #getting undefined reference when I change to Array{Array{}} instead of matrix

    camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)

    for v=1:num_videos
        for f=1:num_frames
            arr = dict[v]["views"][f]["ground_truth"]["labels"]
            center = dict[v]["views"][f]["ground_truth"]["centers"]

            temp = []
            for i = 1:length(arr)
                x = center[i][1]
                y = center[i][2]
                push!(temp, (x, y, arr[i]))
            end
            objects_observed[v, f] = convert(Array{Detection2D}, temp)

            #camera postion
            x = dict[v]["views"][f]["camera"]["x"]
            y = dict[v]["views"][f]["camera"]["y"]
            z = dict[v]["views"][f]["camera"]["z"]
            #focus
            f_x = dict[v]["views"][f]["lookat"]["x"]
            f_y = dict[v]["views"][f]["lookat"]["y"]
            f_z = dict[v]["views"][f]["lookat"]["z"]
            c = Camera_Params(camera_location = Coordinate(x,y,z), camera_focus = Coordinate(f_x,f_y,f_z))
            camera_trajectories[v, f] = c
        end
    end
    return objects_observed, camera_trajectories
end




#dict is actually an array of dictionaries
function write_to_dict(dict::Array{Any,1}, camera_trajectories::Matrix{Camera_Params}, inferred_realities, num_videos::Int64, num_frames::Int64)
    params = Video_Params()

    #add inferences about the objects in the scenes
    for v = 1:num_videos
        #for each object
        #parsed = eval(Meta.parse(inferred_realities[v])) #or eval(Meta.parse(inferred_realities[v][1]))
        a = Array{Any}(undef, length(inferred_realities[v]))
        for i in 1:length(inferred_realities[v])
            r = inferred_realities[v][i]
            a[i] = Dict("label" => r[4], "position" => r[1:3])
        end
        d = dict[v]
        merge!(d, Dict("inferences" => a))

        #add per frame
        for f = 1:num_frames
            labels = []
            centers = []
            for i in 1:length(inferred_realities[v])
                r = inferred_realities[v][i]
                x, y = get_image_xy(camera_trajectories[v,f], params, Coordinate(r[1],r[2],r[3]))
                if within((x, y), Receptive_Field((0,0), (params.image_dim_x, params.image_dim_y)))
                    push!(labels, r[4])
                    push!(centers, (x, y))
                end
            end
            d_f = d["views"][f]
            a = Dict("labels" => labels, "centers" => centers)
            merge!(d_f, Dict("inferences" => a))
        end
    end

    return dict
end

# function count_observations(objects_observed::Matrix{Array{Array{Detection2D}}})
#     num_videos = 100
#     num_frames = 75
#
#     zeros(91)
#
#     arr = []
#
#     for v=1:num_videos
#         arr_per_vid = []
#         for f=1:num_frames
#             n_obj = length(objects_observed[v, f])
#             for i = 1:n
#                 push!(arr_per_vid, objects_observed[v, f][])
#             end
#         end
#     end
#
# end
