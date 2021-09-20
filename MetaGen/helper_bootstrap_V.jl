
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
#returns a dataframe with the MSE for each particle's V
function make_MSE(data::DataFrame, gt_data::DataFrame, num_particles::Int64,
    n_possible_objects::Int64)

    col_names = []
    push!(col_names, "MSE")
    for j = 1:num_particles
        push!(col_names, "MSE_" * string(j))
    end
    data_MSE = DataFrame()
    for i=1:length(col_names)
        data_MSE[:, col_names[i]] = Float64[]
    end

    for v = 1:num_videos

        MSEs = []

        SEs = Float64[]
        for obj = 1:n_possible_objects
            for e = 1:2 #false alarms and misses
                str = e==1 ? "fa_" * string(obj)  : "m_" * string(obj)
                gt_str = "gt_" * str
                push!(SEs, SE(data[v, str], gt_data[v, gt_str]))
            end
        end
        MSE = sum(SEs)/length(SEs)
        push!(MSEs, MSE)


        for j = 1:num_particles
            SEs = Float64[]
            for obj = 1:n_possible_objects
                for e = 1:2 #false alarms and misses
                    str = e==1 ? "fa_" * string(obj) * "_" * string(j) : "m_" * string(obj) * "_" * string(j)
                    gt_str = e==1 ? "gt_fa_" * string(obj) : "gt_m_" * string(obj)
                    push!(SEs, SE(data[v, str], gt_data[v, gt_str]))
                end
            end
            MSE = sum(SEs)/length(SEs)
            push!(MSEs, MSE)
        end
        push!(data_MSE, MSEs)
    end

    return data_MSE
end


#squared error between two numbers
function SE(a::Float64, b::Float64)
    return (a - b)^2
end

################################################################################

#takes two dataframes: the dataframe with weights, and the one with MSEs
#returns a dataframe with three columns: mean, upper, and lower CI for overall MSE of V
function confidence_interval(data::DataFrame, MSE_data::DataFrame, num_particles::Int64,
    n_possible_objects::Int64, n_boot::Int64, cil::Float64)

    data_ci = DataFrame(upper_MSE = Float64[], lower_MSE = Float64[])

    for v = 1:num_videos
        weights = Array{Float64}(undef, num_particles)
        for j = 1:num_particles
            str = "weight_" * string(j)
            weights[j] = data[v, str]
        end
        normalized_log_weights = weights ./ (sum(weights))

        new_row = Float64[] #get bootstrapped values
        values = Float64[]
        for j = 1:num_particles
            str = "MSE_" * string(j)
            push!(values, MSE_data[v, str])
        end
        df = DataFrame(weights = exp.(normalized_log_weights), values = values)
        bs = bootstrap(weighted_average, df, BasicSampling(n_boot))
        bci = confint(bs, PercentileConfInt(cil))
        lower_ci = bci[1][2]
        upper_ci = bci[1][3]
        push!(new_row, lower_ci)
        push!(new_row, upper_ci)

        push!(data_ci, new_row)
    end
    return data_ci
end
