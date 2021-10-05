
#takes a dataframe and returns a dataframe with upper and lower CI
function confidence_interval(data::DataFrame, num_particles::Int64,
    n_possible_objects::Int64, n_boot::Int64, cil::Float64)

    col_names = []
    for obj = 1:n_possible_objects
        push!(col_names, "fa_" * string(obj) * "_lower")
        push!(col_names, "fa_" * string(obj) * "_upper")
        push!(col_names, "m_" * string(obj) * "_lower")
        push!(col_names, "m_" * string(obj) * "_upper")
    end
    data_ci = DataFrame()
    for i=1:length(col_names)
        data_ci[:, col_names[i]] = Float64[]
    end

    for v = 1:num_videos
        weights = Array{Float64}(undef, num_particles)
        for j = 1:num_particles
            str = "weight_" * string(j)
            weights[j] = data[v, str]
        end
        normalized_log_weights = weights ./ (sum(weights))

        new_row = Float64[] #get bootstrapped values
        for obj = 1:n_possible_objects
            for e = 1:2 #false alarms and misses
                values = Float64[]
                for j = 1:num_particles
                    str = e==1 ? "fa_" * string(obj) * "_" * string(j) : "m_" * string(obj) * "_" * string(j)
                    push!(values, data[v, str])
                end
                df = DataFrame(weights = exp.(normalized_log_weights), values = values)
                bs = bootstrap(weighted_average, df, BasicSampling(n_boot))
                bci = confint(bs, PercentileConfInt(cil))
                lower_ci = bci[1][2]
                upper_ci = bci[1][3]
                push!(new_row, lower_ci)
                push!(new_row, upper_ci)
            end
        end
        push!(data_ci, new_row)
    end
    return data_ci
end

#dataframe has a column named "weights" and a column named "values" and returns a weighted average of the values
function weighted_average(data::DataFrame)
    return sum(data[!, "values"] .* (data[!, "weights"]./sum(data[!, "weights"])))
end

################################################################################
#takes a dataframe. returns the MSE of the weighted average V
#each row of the dataframe is a particle. columns are the weight of the particle,
#the fa and m, and the gt_fa, gt_m
function make_MSE(data::DataFrame)

    n_possible_objects = convert(Int64,(size(df)[2]-1)/4)
    num_particles = size(df)[1]

    weights = data[!, "weights"]
    v_matrixes = Array{Matrix{Float64}}(undef, num_particles)
    for j = 1:num_particles
        v_matrix = zeros(params.n_possible_objects, 2)
        for obj = 1:n_possible_objects
            for e = 1:2 #false alarms and misses
                str = e==1 ? "fa_" * string(obj) : "m_" * string(obj)
                v_matrix[obj, e] = data[j, str]
            end
        end
        v_matrixes[j] = v_matrix
    end
    normalized_log_weights = weights ./ (sum(weights))
    avg_v = sum(exp.(normalized_log_weights)./sum(exp.(normalized_log_weights)) .* v_matrixes)#change out of log

    #get the gt_v_matrix
    gt_v_matrix = zeros(params.n_possible_objects, 2)
    for obj = 1:n_possible_objects
        for e = 1:2
            gt_str = e==1 ? "gt_fa_" * string(obj)  : "gt_m_" * string(obj)
            gt_v_matrix[obj, e] = data[1, gt_str] #every row should have the same thing for gt, so just take the first one
        end
    end

    #now calculate the MSE for this avg_v
    return sum((avg_v .- gt_v_matrix).^2) / length(gt_v_matrix)
end


################################################################################

#takes two dataframes: the dataframe with particles, and the ground truth
#returns a dataframe with three columns: mean, upper, and lower CI for overall MSE of V
function MSE_and_confidence_interval(data::DataFrame, gt_data::DataFrame, num_particles::Int64,
    n_possible_objects::Int64, n_boot::Int64, cil::Float64)

    data_ci = DataFrame(MSE = Float64[], upper_MSE = Float64[], lower_MSE = Float64[])

    for v = 1:num_videos

        #set up the dataframe where each row is a particle
        col_names = []
        push!(col_names, "weights")
        for obj = 1:n_possible_objects
            push!(col_names, "fa_" * string(obj))
            push!(col_names, "m_" * string(obj))
            push!(col_names, "gt_fa_" * string(obj))
            push!(col_names, "gt_m_" * string(obj))
        end
        df = DataFrame()
        for i=1:length(col_names)
            df[:, col_names[i]] = Float64[]
        end

        for j = 1:num_particles
            new_row = []
            str = "weight_" * string(j)
            push!(new_row, data[v, str])
            for obj = 1:n_possible_objects
                str = "fa_" * string(obj) * "_" * string(j)
                push!(new_row, data[v, str])
                str = "m_" * string(obj) * "_" * string(j)
                push!(new_row, data[v, str])
                str = "gt_fa_" * string(obj)
                push!(new_row, gt_data[v, str])
                str = "gt_m_" * string(obj)
                push!(new_row, gt_data[v, str])
            end
            push!(df, new_row)
        end

        new_row_ci = []
        MSE = make_MSE(df)
        push!(new_row_ci, MSE)
        bs = bootstrap(make_MSE, df, BasicSampling(n_boot))
        bci = confint(bs, PercentileConfInt(cil))
        lower_ci = bci[1][2]
        upper_ci = bci[1][3]
        push!(new_row_ci, lower_ci)
        push!(new_row_ci, upper_ci)

        push!(data_ci, new_row_ci)
    end
    return data_ci
end
