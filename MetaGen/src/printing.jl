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
#does weighted average of V, but highest weighted particle for world state. also prints world state and weight for each particle
function print_Vs_and_Rs_to_file(V_file, ws_file, traces, num_particles::Int64, params::Video_Params, v::Int64, last_time::Bool=false)
    print(V_file, v, " & ")
    print(ws_file, v, " & ")

    avg_v, world_states, best_world_state = process(traces, num_particles, params, v)

    for i = 1:params.n_possible_objects
        print(V_file, avg_v[i,1], "&")
        print(V_file, avg_v[i,2], "&")
    end

    v_matrix = zeros(params.n_possible_objects, 2)
    for j = 1:num_particles
        choices = get_choices(traces[j])
        weight = get_score(traces[j])
        for i = 1:params.n_possible_objects
            v_matrix[i,1] = choices[:videos => v => :v_matrix => :lambda_fa => i => :fa]
            v_matrix[i,2] = choices[:videos => v => :v_matrix => :miss_rate => i => :miss]
            print(V_file, v_matrix[i,1], "&")
            print(V_file, v_matrix[i,2], "&")
        end
        print(V_file, weight, "&")
    end


    print(V_file, best_world_state)
    ###########################################
    #switch to ws file
    print(ws_file, best_world_state, "&")

    for j = 1:num_particles
        choices = get_choices(traces[j])
        world_state = choices[:videos => v => :init_scene]
        weight = get_score(traces[j])
        if j==num_particles && last_time #if this is the last thing printing, don't put &
            print(ws_file, world_state, "&")
            print(ws_file, weight)
        else
            print(ws_file, world_state, "&")
            print(ws_file, weight, "&")
        end
    end
    print(V_file, "\n")
    print(ws_file, "\n")
    return (best_world_state, avg_v) #return the mode reality and average v
end

##############################################################################################
#returns the weighted average v_matrix and the highest-weighted world state for a set particles
function process(traces, num_particles::Int64, params::Video_Params, v::Int64)

    weights = Array{Float64}(undef, num_particles)
    v_matrixes = Array{Matrix{Float64}}(undef, num_particles)
    world_states = Array{Array{Any,1}}(undef, num_particles)

    for i = 1:num_particles
        weights[i] = get_score(traces[i])
        choices = get_choices(traces[i])
        v_matrix = zeros(params.n_possible_objects, 2)
        for j = 1:params.n_possible_objects
            v_matrix[j,1] = choices[:videos => v => :v_matrix => :lambda_fa => j => :fa]
            v_matrix[j,2] = choices[:videos => v => :v_matrix => :miss_rate => j => :miss]
        end
        v_matrixes[i] = v_matrix
        world_states[i] = choices[:videos => v => :init_scene]
    end
    normalized_log_weights = weights ./ (sum(weights))
    avg_v = sum(exp.(normalized_log_weights)./sum(exp.(normalized_log_weights)) .* v_matrixes)#change out of log
    best_world_state = world_states[findmax(weights)[2]]

    return avg_v, world_states, best_world_state
end

export print_Vs_and_Rs_to_file_new
export countmemb
