#Main script for set up and inference (with lesion) using the generative model
#include("gm_Julian_intermediate.jl")
include("gm_with_FA_M.jl")
include("inference_with_real_input.jl")
include("shared_functions.jl") #just want countmemb function
include("read_output_from_detectron.jl")

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs

#creating output file
#outfile = string("output", ARGS[1], ".csv")
outfile = string("output1.csv")
file = open(outfile, "w")

#Defining observations / constraints
possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#each V sill have n_percepts, that many movies
#n_percepts = 100 #particle filter is set up such that it needs at least 2 percepts
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
print(file, "time elapsed PF & num_particles & num_samples & num_moves & avg_V PF & frequency table Vs after PF & frequency table PF & time elapsed lesioned PF & avg_V lesioned PF & frequency table Vs after lesioned PF & frequency table lesioned PF \n")

##########################

#files with input to read
files = ["Detectron2output_file.txt", "Detectron2output_file2.txt", "Detectron2output_file3.txt"]
n_percepts = length(files)
gt_percepts = read_output_from_detectron(files)

#something like
gt_R = [["person"], ["person"], ["person"]]
gt_reality = gt_R
gt_V = calculate_V(gt_percepts, gt_reality)
#gt_V = Matrix{Float64}(undef, length(possible_objects), 2)
#########################

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

num_particles = 100 #100

#num_samples to return
num_samples = 100 #100

#num perturbation moves
num_moves = 1

#(traces,time_particle) = @timed particle_filter(num_particles, gt_percept, num_samples);
#println(file, "time elaped \n ", time_particle)
(traces, time_PF) = @timed particle_filter(num_particles, gt_percepts, num_samples, beta_mean, false);
print(file, "time elapsed particle filter  ", time_PF, " & ")

print(file, num_particles, " & ")
print(file, num_samples, " & ")
print(file, num_moves, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

#############################################

#execute lesioned inference procedure
(traces, time_PF) = @timed particle_filter(num_particles, gt_percepts, num_samples, beta_mean, true);

print(file, "time elapsed lesioned particle filter  ", time_PF, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects, true)

close(file)
