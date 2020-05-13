#Main script for executing data simulation and inference using the generative model
include("gm_Mario_model.jl")
include("inference_Mario_model.jl")
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
n_frames = 2


#file header
print(file, "gt_V & gt_R & ")
for p=1:n_percepts
	print(file, "percept", p, " & ")
end
for p=1:n_percepts
	print(file, "avg V after p", p, " & ")
	print(file, "frequency table PF after p", p, " & ")
end
println(file, "time elapsed PF & num_particles & num_samples & num_moves & frequency table PF & unique_realities PF & mode realities PF & avg_Vs_binned for unique_realities PF & avg_Rs PF & Euclidean distance between avg_Rs and gt_R PF & Euclidean distance FA PF & Euclidean distance M PF &")

##########################

##For Simulated data

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames))
gt_reality, gt_locationses, gt_V, gt_percepts = Gen.get_retval(gt_trace)
gt_choices = get_choices(gt_trace)

println("gt_choices")
display(gt_choices)

#will have to find a way to generate gt_choices from gt_percepts
#for when I'm no longer using simulated data




##########################

##The files are output from Detectron2. They contian the COCO coding of the object categories

# #list of files to read
# files = ["Bicyclist", "Motorcycle-car", "Toddler-bicycle"]


# #get the percepts in number form. Each percept/video is a file, each line is a frame
# gt_percepts = []
# for p = 1:n_percepts

# 	#open input file
#     #input_file = open(files[p])
#     #slurp = read(input_file, String)

#     open(files[p]) do file2
# 	    percept = []

#         #each percept should have the number of frames specified in the GM
#         @assert countlines(files[p]) == n_frames


#         for line in enumerate(eachline(file2))

#             #line is a tuple consisting of the line number and the string
#             #changing to just the string
#             line=line[2]

#       		start = findfirst("[",line)[1]
#       		finish = findfirst("]",line)[1]
#       		pred = line[start+1:finish-1]
#       		arr = split(pred, ", ")
#       		frame = Array{Int}(undef, length(arr))
#       		for j = 1:length(arr)
#       		    #add 1 to fix indexing discrepancy between python indices for COCO dataset and Julia indexing
#       		    frame[j] = parse(Int,arr[j])+1
#       		end
#       		push!(percept,frame)
#         end

#         push!(gt_percepts,percept)
#     end
# end


#########################


gt_R_bool = names_to_boolean
print(file, gt_V, " & ")
print(file, gt_reality, " & ")

#println("gt_percepts", gt_percepts)

#Saving gt_percepts to file
percepts = []
for p = 1:n_percepts
	percept = []
	#println("gt_percepts[p] ", gt_percepts[p])
	for f = 1:n_frames
		#println("gt_percepts[p][f] ", gt_percepts[p][f])
		#println("possible_objects[gt_percepts[p][f]] ", possible_objects[gt_percepts[p][f]])
		perceived_frame = gt_percepts[p][f]
		print(file,  perceived_frame)
	end
	print(file, " & ")
end

#println("percepts ", gt_percepts)


# #store the visual counts in the submap to observations
# #not sure this is even used
# observations = Gen.choicemap()
# for p = 1:n_percepts
# 	for f = 1:n_frames
# 		#observations[(:perceived_frame,p,f)] = gt_percepts[p][f]
# 		addr = (:perceived_frame,p,f) => :visual_count
# 		sm = Gen.get_submap(gt_choices, addr)
# 		Gen.set_submap!(observations, addr, sm)
# 	end
# end

# ##############################################

#Perform particle filter

num_particles = 100 #100

#num_samples to return
num_samples = 100 #100

#num perturbation moves
num_moves = 1

#(traces,time_particle) = @timed particle_filter(num_particles, gt_percept, num_samples);
#println(file, "time elaped \n ", time_particle)
(traces, time_PF) = @timed particle_filter(num_particles, gt_percepts, gt_choices, num_samples);
print(file, "time elapsed particle filter  ", time_PF, " & ")

print(file, num_particles, " & ")
print(file, num_samples, " & ")
print(file, num_moves, " & ")


#extract results
realities = Array{Array{String}}[]
Vs = Array{Float64}[]
for i = 1:num_samples
	Rs,V,_ = Gen.get_retval(traces[i])
	push!(Vs,V)
	push!(realities,Rs)
end

#############################################
#Analysis function needs the realities and Vs resulting from a sampling procedure and gt_V
function analyze(realities, Vs, gt_V, gt_R)

	#want to make a frequency table of the realities sampled
	ft = freqtable(realities)
	println(ft)
	dictionary = countmemb(realities)
	print(file ,dictionary, " & ")

	#want, for each reality, to bin Vs
	unique_realities = unique(realities)

	unique_Vs = unique(Vs)

	avg_Vs_binned = Array{Float64}[]
	freq = Array{Float64}(undef, length(unique_realities))

	how_many_unique = length(unique_realities)
	for j = 1:how_many_unique
		index = findall(isequal(unique_realities[j]),realities)
		#freq keeps track of how many there are
		freq[j] = length(index)
		push!(avg_Vs_binned, mean(Vs[index]))
	end


	#find avg_Vs_binned at most common realities and compute euclidean distances
	#index of most frequent reality. does not work with ties.
	idx = findfirst(isequal(maximum(freq)),freq)
	unique_realities[idx]
	print(file, unique_realities, " & ")
	#mode reality
	print(file, unique_realities[idx], " & ")
	print(file, avg_Vs_binned, " & ")


	#average Rs in boolean format.
	n_percepts = size(gt_R)[1]
	#avg_Rs will keep track of the avg_R for each percept
	avg_Rs = []
	for p = 1:n_percepts
		avg_R = zeros(length(possible_objects))
		for j = 1:how_many_unique
			total = get(dictionary,string(unique_realities[j]),0)
			avg_R = avg_R + names_to_boolean(unique_realities[j][p],possible_objects)*total/length(realities)
		end
		push!(avg_Rs, avg_R)
	end

	print(file, avg_Rs, " & ")

	#take distance between avg_Rs and gt_R
	distances = []
	gt_R_bools = []
	for p = 1:n_percepts
		gt_R_bool = names_to_boolean(gt_R[p],possible_objects)
		push!(gt_R_bools, gt_R_bool)
		push!(distances, euclidean(avg_Rs[p], gt_R_bool))
	end
	print(file, distances, " & ")

	#compare mean V of most frequent reality to gt_V
	#for false alarms
	dist_FA = euclidean(gt_V[1], avg_Vs_binned[idx][1])
	print(file, dist_FA, " & ")
	#for miss rates
	dist_M = euclidean(gt_V[2], avg_Vs_binned[idx][2])
	print(file, dist_M)
	#need to add & after calling analyze the first time

end;

#############################################

analyze(realities, Vs, gt_V, gt_reality)
close(file)
