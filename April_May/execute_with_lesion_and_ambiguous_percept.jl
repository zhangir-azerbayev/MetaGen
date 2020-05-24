#Main script for executing data simulation and inference using the generative model
#include("gm_Julian_intermediate.jl")
include("gm_with_FA_M.jl")
include("inference_with_lesioned_option_and_ambiguous_percept.jl")
include("shared_functions.jl") #just want countmemb function

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs

#creating output file
outfile = string("output", ARGS[1], ".csv")
#outfile = string("output111.csv")
file = open(outfile, "w")

#Defining observations / constraints
possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#each V sill have n_percepts, that many movies
n_percepts = 50 #particle filter is set up such that it needs at least 2 percepts
n_frames = 10


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

#unlesioned model
#for ambiguous_percept
print(file, "avg V after ambiguous_percept & ")
print(file, "frequency table Vs ambiguous_percept & ")
print(file, "frequency table PF ambiguous_percept & ")
#unlesioned model
print(file, "time elapsed PF & num_particles & num_samples & num_moves & avg_V PF & frequency table Vs after PF & frequency table PF & ")

#lesioned model
#for ambiguous_percept
print(file, "lesioned model avg V after ambiguous_percept & ")
print(file, "lesioned model frequency table Vs ambiguous_percept & ")
print(file, "lesioned model frequency table PF ambiguous_percept & ")
#lesioned model
print(file, "time elapsed lesioned PF & avg_V lesioned PF & frequency table Vs after lesioned PF & frequency table lesioned PF & ambiguous_percept \n")
##########################

##For Simulated data

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames))
gt_reality,gt_V,gt_percepts = Gen.get_retval(gt_trace)
gt_choices = get_choices(gt_trace)

println("gt_choices")
display(gt_choices)

#ambiguous percept: percept 51
ambiguous_percept = [["motorcycle"],["motorcycle"],["motorcycle"],["motorcycle"],["motorcycle"],["motorcycle", "airplane"],["motorcycle", "airplane"],["motorcycle", "airplane"],["motorcycle", "airplane"],["motorcycle", "airplane"]]




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

#parameters for truncated normal for FA and M
alpha = 2
beta = 10
beta_mean = alpha / (alpha + beta)

#Perform particle filter

num_particles = 200 #100

#num_samples to return
num_samples = 200 #100

#num perturbation moves
num_moves = 1

#(traces,time_particle) = @timed particle_filter(num_particles, gt_percept, num_samples);
#println(file, "time elaped \n ", time_particle)
(traces, time_PF) = @timed particle_filter(num_particles, gt_percepts, ambiguous_percept, gt_choices, num_samples, beta_mean, false);
print(file, "time elapsed particle filter  ", time_PF, " & ")

print(file, num_particles, " & ")
print(file, num_samples, " & ")
print(file, num_moves, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

#############################################

#execute lesioned inference procedure
(traces, time_PF) = @timed particle_filter(num_particles, gt_percepts, ambiguous_percept, gt_choices, num_samples, beta_mean, true);

print(file, "time elapsed lesioned particle filter  ", time_PF, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

print(file, ambiguous_percept)

close(file)
