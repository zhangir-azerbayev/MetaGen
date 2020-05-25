#Main script for executing data simulation and inference using the generative model

#After inferring a V, run inference procedure again conditioning on V
#There are no ambiguous percepts in this one
#This version accomodates a variable number of frames

include("gm_with_FA_M_with_variable_frames.jl")
include("inference_with_fully_trained_metagen.jl")
include("shared_functions.jl") #just want countmemb function

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs

#creating output file
#outfile = string("output", ARGS[1], ".csv")
outfile = string("output111.csv")
file = open(outfile, "w")

#Defining observations / constraints
possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#each V sill have n_percepts, that many movies
n_percepts = 3 #particle filter is set up such that it needs at least 2 percepts

#uniformly between 5 and 15 frames per percept
n_frames = convert(Array{Int,1}, floor.(11*(rand(Float64, 50)).+5))

#file header
print(file, "gt_V & gt_R & ")
for p=1:n_percepts
	print(file, "percept", p, " & ")
end
for p=1:n_percepts
	print(file, "avg V after p", p, " & ")
	print(file, "frequency table Vs after p", p, " & ")
	print(file, "frequency table PF after p", p, " & ")
end

#online model
print(file, "time elapsed PF & num_particles & num_samples & num_moves & avg_V PF & frequency table Vs after PF & frequency table PF & ")

#retrospective model
print(file, "time elapsed retrospective PF & avg_V retrospective PF & frequency table Vs after retrospective PF & frequency table retrospective PF & ")

#lesioned model
print(file, "time elapsed lesioned PF & avg_V lesioned PF & frequency table Vs after lesioned PF & frequency table lesioned PF \n")
##########################

##For Simulated data

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames))
gt_reality,gt_V,gt_percepts = Gen.get_retval(gt_trace)
gt_choices = get_choices(gt_trace)

println("gt_choices")
display(gt_choices)


gt_R_bool = names_to_boolean
print(file, gt_V, " & ")
print(file, gt_reality, " & ")

#println("gt_percepts", gt_percepts)

#Saving gt_percepts to file
percepts = []
for p = 1:n_percepts
	percept = []
	n_frames_in_this_percept = n_frames[p]
	#println("gt_percepts[p] ", gt_percepts[p])
	for f = 1:n_frames_in_this_percept
		#println("gt_percepts[p][f] ", gt_percepts[p][f])
		#println("possible_objects[gt_percepts[p][f]] ", possible_objects[gt_percepts[p][f]])
		perceived_frame = gt_percepts[p][f]
		print(file,  perceived_frame)
	end
	print(file, " & ")
end

#############################################
#Analysis function needs the realities and Vs resulting from a sampling procedure and gt_V
function analyze(realities)

	#want to make a frequency table of the realities sampled
	ft = freqtable(realities)
	println(ft)
	dictionary = countmemb(realities)
	print(file ,dictionary, " & ")

end;

#############################################
#Perform particle filters

num_particles = 100

#num_samples to return
num_samples = 100

#num perturbation moves
num_moves = 1

#############################################
#Online model

#garbage, won't be used since lesion is false
V = Matrix{Float64}(undef, length(possible_objects), 2)

(traces, time_PF) = @timed particle_filter(num_particles, n_percepts, n_frames, gt_choices, num_samples, V, false);
print(file, "time elapsed particle filter  ", time_PF, " & ")

print(file, num_particles, " & ")
print(file, num_samples, " & ")
print(file, num_moves, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

#############################################
#lesion with learned V
Vs = Array{Float64}[]
for i = 1:num_samples
	_,V,_ = Gen.get_retval(traces[i])
	push!(Vs, V)
end
#maps from V to frequency
dictionary_Vs = countmemb(Vs)
#invert the mapping
frequency_Vs = Dict()
for (k, v) in dictionary_Vs
    if haskey(frequency_Vs, v)
        push!(frequency_Vs[v],k)
    else
        frequency_Vs[v] = [k]
    end
end

arr = collect(keys(frequency_Vs))
arr_as_numeric = convert(Array{Int64,1}, arr)
m = maximum(arr_as_numeric) #finding mode V
#length(frequency_Vs[m])==1 ? V = frequency_Vs[m] : V = frequency_Vs[m][1] #in case of tie, take the first V
V_as_string = frequency_Vs[m][1]
#parsing string. remove comma and semicolon, and brackets
V_as_string = replace.(V_as_string, r"[,;]" => "")
V_as_string = chop(V_as_string, head=1, tail=1)
V_as_array = [parse(Float64, ss) for ss in split(V_as_string)]

V = Matrix{Float64}(undef, length(possible_objects), 2)
k = Int64
k=0
for i = 1:size(V)[1]
	for j = 1:size(V)[2]
		global k += 1
		V[i,j] = V_as_array[k]
	end
end

(traces, time_PF) = @timed particle_filter(num_particles, n_percepts, n_frames, gt_choices, num_samples, V, true);
print(file, "time elapsed lesioned particle filter  ", time_PF, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

#############################################
#lesioned model
#parameters prior over FA and M
alpha = 2
beta = 10
beta_mean = alpha / (alpha + beta)

V = Matrix{Float64}(undef, length(possible_objects), 2)
fill!(V, beta_mean)

(traces, time_PF) = @timed particle_filter(num_particles, n_percepts, n_frames, gt_choices, num_samples, V, true);

print(file, "time elapsed lesioned particle filter  ", time_PF, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

close(file)
