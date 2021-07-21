
function make_observations(dict::Array{Any,1}, receptive_fields::Vector{Receptive_Field})

    num_videos = 2
    num_frames = 300

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
            labels = dict[v]["views"][f]["detections"]["labels"][indices]
            center = dict[v]["views"][f]["detections"]["center"][indices]
            temp = Array{Detection2D}(undef, length(labels))
            for i = 1:length(labels)
                label = labels[i]
                x = center[i][1]
                y = center[i][2]
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
