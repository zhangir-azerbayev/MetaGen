#script for making visualizations

function visualize_observations(objects_observed::Matrix{Array{Array{Detection2D}}}, v::Int64, f::Int64, receptive_fields)

    xs = []
    ys = []
    cs = []
    for rf = 1:length(receptive_fields)
        objects = objects_observed[v, f][rf]
        if length(objects) > 0
            for i = 1:length(objects)
                obj = objects[i]
                push!(xs, obj[1])
                push!(ys, obj[2])
                push!(cs, obj[3])
            end
        end
    end

    if length(xs) < 1
        push!(xs, Inf)
        push!(ys, Inf)
        push!(cs, 1000000)
    end

    println("xs", xs)
    println("ys", ys)
    pyplot()
    p = scatter(xs, ys, color = cs, series_annotations = text.(cs, :bottom), title = "Observations/Detections", xlims = (0, 360), ylims = (0, 240))
    display(p)
    savefig(p, "observations.pdf")
end

#j is which trace
function visualize_trace(traces, j::Int64, camera_trajectories::Matrix{Camera_Params}, v::Int64, f::Int64, params::Video_Params)
    inferred_scene = traces[j][:videos => v => :init_scene]
    println("inferred_scene ", inferred_scene)

    #figure out where each object in the inferred_scene scene would show up in frame f
    xs = []
    ys = []
    cs = []
    if length(inferred_scene) > 0
        for i = 1:length(inferred_scene)
            obj = inferred_scene[i]
            (x,y) = get_image_xy(camera_trajectories[v, f], params, Coordinate(obj[1], obj[2], obj[3]))
            push!(xs, x)
            push!(ys, y)
            push!(cs, obj[4])
        end
    else
        push!(xs, Inf)
        push!(ys, Inf)
        push!(cs, 1000000)
    end

    #how many objects are out-of-view? total - number objects in-view
    in_x = (xs .<= 360) .& (xs .>= 0)
    in_y = (ys .<= 240) .& (ys .>= 0)
    num_obj_out_of_view = length(inferred_scene) - sum(in_x .& in_y)


    p = plt.figure()
    plt.scatter(xs, ys)
    for (i, c) in enumerate(cs)
        plt.annotate(c, (xs[i], ys[i]))
    end
    plt.xlim([0, 360])
    plt.ylim([0, 240])
    p.suptitle("particle $(j)'s inferences. weight: $(get_score(traces[j])). num obj out of view: $num_obj_out_of_view")

    #add observations. hard-coded.
    # circle1 = plt.Circle((160, 120), 30, color='r')
    # circle2 = plt.Circle((165, 125), 30, color='b')
    # fig = plt.gcf()
    # ax = fig.gca()
    # ax.add_patch(circle1)
    # ax.add_patch(circle2)
    # println("here")

    p.savefig("v $(v) particle $(j) .pdf")
end

function visualize_trace_with_heatmap(traces, camera_trajectories::Matrix{Camera_Params}, v::Int64, f::Int64, params::Video_Params)
    xs = []
    ys = []
    cs = []
    weights =[]

    for (j, trace) in enumerate(traces)
        inferred_scene = trace[:videos => v => :init_scene]
        for obj in inferred_scene
            (x, y) = get_image_xy(camera_trajectories[v, f], params, Coordinate(obj[1], obj[2], obj[3]))
            push!(xs, x)
            push!(ys, y)
            push!(cs, obj[4])
            push!(weights, get_score(trace))
        end
        visualize_scene_3d(inferred_scene, "inference_particle_$(j)", "inference_particle_$(j).pdf")
    end

    # Makes heat map that ignores object categories
    p = plt.figure()
    plt.xlim([0, 360])
    plt.ylim([0, 240])
    bins = plt.hexbin(xs, ys, C=weights, gridsize=30, cmap="Greens")
    plt.suptitle("Video $(v)")
    p.savefig("heatmap.pdf")
    plt.close()

    # Makes heat map for each object category
    for cat in Set(cs)
        cat_xs = [x for (x, c) in zip(xs, cs) if c==cat]
        cat_ys = [y for (y, c) in zip(ys, cs) if c==cat]
        cat_weights = [weight for (weight, c) in zip(weights, cs) if c==cat]

        q = plt.figure()
        plt.xlim([0, 360])
        plt.ylim([0, 240])
        plt.hexbin(cat_xs, cat_ys, C=cat_weights, gridsize=10, cmap="Greens")
        q.suptitle("Video $(v) Object $(cat)")
        q.savefig("heatmap_cat$(cat).pdf")
        plt.close()
    end
end

function visualize_scene_3d(scene, title, filename)
    xs = []
    ys = []
    zs = []
    cs = []

    for obj in scene
        push!(xs, obj[1])
        push!(ys, obj[2])
        push!(zs, obj[3])
        push!(cs, obj[4])
    end

    p = plt.figure()
    ax = p.add_subplot(projection="3d")
    ax.scatter(xs, ys, zs)
    ax.set_xlim(-16, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    for (i, c) in enumerate(cs)
        ax.text(xs[i], ys[i], zs[i], c)
    end
    ax.set_title(title)

    p.savefig(filename)
end

export visualize_observations
export visualize_trace
export visualize_trace_with_heatmap
export visualize_scene_3d
