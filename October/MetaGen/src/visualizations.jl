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
        push!(cs, Inf)
    end

    pyplot()
    p = scatter(xs, ys, color = cs, series_annotations = text.(cs, :bottom), title = "particle $(j)'s inferences. weight: $(get_score(traces[j]))", xlims = (0, 360), ylims = (0, 240))
    display(p)
    savefig(p, "v $(v) particle $(j) .pdf")
end


export visualize_observations
export visualize_trace
