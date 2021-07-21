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

#function for printing stuff to file
#needs traces, num_samples, r for which reality to extract
function print_Vs_and_Rs_to_file(file, tr, num_samples::Int64, params, r::Int64, last_time::Bool=false)
    #initialize something for tracking average V
    avg_V = zeros(length(params.possible_objects), 2)
    #initialize something for realities
    realities = Array{Array{Any,1}}(undef, num_samples)
    avg_v = zeros(Float64, length(params.possible_objects), 2)
    for i = 1:num_samples
        choices = get_choices(tr[i])
        #extract v
        for j = 1:length(params.possible_objects)
            avg_v[j,1] = avg_v[j,1] + choices[:videos => r => :v_matrix => :lambda_fa => j => :fa]/num_samples
            avg_v[j,2] = avg_v[j,2] + choices[:videos => r => :v_matrix => :miss_rate => j => :miss]/num_samples
        end
        #extract r
        #println("r ", r)
        realities[i] = choices[:videos => r => :init_scene]
    end
    # println("avg_V is ", avg_V)
    print(file, avg_v, " & ")

    # dictionary_Vs = countmemb(Vs)
    # print(file, dictionary_Vs, " & ")

    #instead of printing dictionary of realities, just print the mode
    dictionary_realities = countmemb(realities)
    print(file, dictionary_realities, " & ")
    #invert the mapping
    frequency_realities = Dict()
    for (k, v) in dictionary_realities
        if haskey(frequency_realities, v)
            push!(frequency_realities[v],k)
        else
            frequency_realities[v] = [k]
        end
    end

    arr = collect(keys(frequency_realities))
    arr_as_numeric = convert(Array{Int64,1}, arr)
    m = maximum(arr_as_numeric) #finding mode
    #length(frequency_Vs[m])==1 ? V = frequency_Vs[m] : V = frequency_Vs[m][1] #in case of tie, take the first V
    reality_as_string = frequency_realities[m][1]

    if last_time #if this is the last thing printing, don't put &
        print(file ,reality_as_string)
    else
        print(file ,reality_as_string, " & ")
    end

    return frequency_realities[m] #return the mode reality
end

export print_Vs_and_Rs_to_file
export countmemb
