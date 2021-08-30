##############################################################################################
#Setting up helper functions
function countmemb(itr)
    d = Dict{String, Int}()
    for val in itr
        if isa(val, Number) && isnan(val)
            continue
        end
        d[string(val)] = get!(d, string(val), 0) + 1
    end
    return d
end

##############################################################################################
#does weighted average of V, but highest weighted particle for world state.
function print_Vs_and_Rs_to_file_new(file, traces, num_samples::Int64, params, v::Int64, last_time::Bool=false)
    print(file, v, " & ")

    weights = Array{Float64}(undef, num_samples)
    v_matrixes = Array{Matrix{Float64}}(undef, num_samples)
    world_states = Array{Array{Any,1}}(undef, num_samples)

    for i = 1:num_samples
        weights[i] = get_score(traces[i])
        choices = get_choices(traces[i])
        v_matrix = zeros(params.n_possible_objects, 2)
        for j = 1:params.n_possible_objects
            v_matrix[j,1] = choices[:videos => v => :v_matrix => :lambda_fa => j => :fa]
            v_matrix[j,2] = choices[:videos => v => :v_matrix => :miss_rate => j => :miss]
        end
        v_matrixes[i] = v_matrix
        world_states[i] = choices[:videos => v => :init_scene]

        println("particle ", i)
        println("weight ", weights[i])
        println("world_state ", world_states[i])
    end

    avg_v = sum(weights./sum(weights) .* v_matrixes)
    for i = 1:params.n_possible_objects
        print(file, avg_v[i,1], "&")
        print(file, avg_v[i,2], "&")
    end

    print(file, world_states, "&")

    best_world_state = world_states[findmax(weights)[2]] #world state with highest weight

    println("max index ", findmax(weights)[2])

    if last_time #if this is the last thing printing, don't put &
        print(file, best_world_state)
    else
        print(file, best_world_state, "&")
        print(file, "\n")
    end

    return (best_world_state, avg_v) #return the mode reality and average v
end

export print_Vs_and_Rs_to_file_new
export countmemb
