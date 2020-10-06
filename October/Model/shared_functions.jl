#This file provides support functions called by the main script execute_for_NEURIPS.jl

using Gen

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
#needs traces, num_samples, possible_objects
function print_Vs_and_Rs_to_file(tr, num_samples, possible_objects, last_time::Bool=false)
    #initialize something for tracking average V
    avg_V = zeros(length(possible_objects), 2)
    #initialize something for realities
    realities = Array{Array{String}}[]
    Vs = Array{Float64}[]
    for i = 1:num_samples
        R,V,_ = Gen.get_retval(tr[i])
        avg_V = avg_V + V/num_samples
        push!(Vs, V)
        push!(realities,R)
        # println("R is ", R)
        # println("V is ", V)
    end
    # println("avg_V is ", avg_V)
    print(file, avg_V, " & ")

    # dictionary_Vs = countmemb(Vs)
    # print(file, dictionary_Vs, " & ")

    #instead of printing dictionary of realities, just print the mode
    dictionary_realities = countmemb(realities)
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
end

##############################################################################################
#Distributions
#TruncatedPoisson
export trunc_poisson

struct TruncatedPoisson <: Gen.Distribution{Int} end

const trunc_poisson = TruncatedPoisson()

function Gen.logpdf(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U) where {U <: Real}
	d = Distributions.Poisson(lambda)
	td = Distributions.Truncated(d, low, high)
	Distributions.logpdf(td, x)
end

function Gen.logpdf_grad(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U)  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::TruncatedPoisson, lambda::U, low::U, high::U)  where {U <: Real}
	d = Distributions.Poisson(lambda)
	rand(Distributions.Truncated(d, low, high))
end

(::TruncatedPoisson)(lambda, low, high) = random(TruncatedPoisson(), lambda, low, high)
is_discrete(::TruncatedPoisson) = true

has_output_grad(::TruncatedPoisson) = false
has_argument_grads(::TruncatedPoisson) = (false,)

##############################################################################################
#TruncatedNormal
export trunc_normal

#small issue where all of the inputs need to by Float64. Doesn't accept Int64s
struct TruncatedNormal <: Gen.Distribution{Float64} end

const trunc_normal = TruncatedNormal()

function Gen.logpdf(::TruncatedNormal, x::U, mu::U, std::U, low::U, high::U) where {U <: Real}
	n = Distributions.Normal(mu, std)
	tn = Distributions.Truncated(n, low, high)
	Distributions.logpdf(tn, x)
end

function Gen.logpdf_grad(::TruncatedNormal, x::U, mu::U, std::U, low::U, high::U)  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::TruncatedNormal, mu::U, std::U, low::U, high::U)  where {U <: Real}
	n = Distributions.Normal(mu, std)
	rand(Distributions.Truncated(n, low, high))
end

(::TruncatedNormal)(mu, std, low, high) = random(TruncatedNormal(), mu, std, low, high)
is_discrete(::TruncatedNormal) = false
has_output_grad(::TruncatedPoisson) = false
has_argument_grads(::TruncatedPoisson) = (false,)

##############################################################################################
#Multivariate uniform
import Gen: random

export mvuniform

struct MultivariateUniform <: Gen.Distribution{Vector{Float64}} end

const mvuniform = MultivariateUniform()

function Gen.logpdf(::MultivariateUniform, z::AbstractVector{T}, low_x::Real, low_y::Real, high_x::Real, high_y::Real) where {T <: Real}
    log_prob_x = (z[1] >= low_x && z[1] <= high_x) ? -log(high_x-low_x) : -Inf
    log_prob_y = (z[2] >= low_y && z[2] <= high_y) ? -log(high_y-low_y) : -Inf
    return log_prob_x+log_prob_y
end

function Gen.logpdf_grad(::MultivariateUniform, x::AbstractVector{T}, low_x::Real, low_y::Real, high_x::Real, high_y::Real) where {T <: Real}
    gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::MultivariateUniform, low_x::Real, low_y::Real, high_x::Real, high_y::Real)
    z = Vector{Float64}(undef, 2)
    z[1] = rand() * (high_x - low_x) + low_x
    z[2] = rand() * (high_y - low_y) + low_y
    return z
end

(::MultivariateUniform)(low_x, low_y, high_x, high_y) = random(MultivariateUniform(), low_x, low_y, high_x, high_y)
has_output_grad(::MultivariateNormal) = false
has_argument_grads(::MultivariateNormal) = (false, false)
