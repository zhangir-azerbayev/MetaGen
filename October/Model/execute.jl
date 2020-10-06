#Main script for executing data simulation and inference using the generative model.
#Running this file simulates one visual system and n_percepts observations. In the NEURIPS paper, we ran this 35000 times to simulate 35000 visual systems.

#In line 19, name the outfile whatever you'd like
#n_percepts (line 23) specifies the number of observations

include("gm_for_NEURIPS.jl")
include("inference_for_NEURIPS.jl")
include("shared_functions_for_NEURIPS.jl")

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs

#creating output file
#outfile = string("output", ARGS[1], ".csv")
outfile = string("output111.csv")
file = open(outfile, "w")

#n_percepts specifies the number of observations (collections of percepts) that the visual system will see
n_percepts = 75 #particle filter is set up such that it needs at least 2 percepts

#uniformly between 5 and 15 frames (percepts) per observation
n_frames = convert(Array{Int,1}, floor.(11*(rand(Float64, n_percepts)).+5))

possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#file header
print(file, "gt_V & gt_R & ")
for p=1:n_percepts
	print(file, "percept", p, " & ")
end
for p=0:n_percepts
	print(file, "online avg V after p", p, " & ")
	print(file, "online mode realities PF after p", p, " & ")
end

#online model
print(file, "time elapsed PF & num_particles & num_samples & num_moves & avg_V PF & mode realities PF & ")

#retrospective model
for p=0:n_percepts
	print(file, "retrospective avg V after p", p, " & ")
	print(file, "retrospective mode realities PF after p", p, " & ")
end
print(file, "time elapsed retrospective PF & avg_V retrospective PF & mode realities retrospective PF & ")

#lesioned model
for p=0:n_percepts
	print(file, "lesioned avg V after p", p, " & ")
	print(file, "lesioned mode realities PF after p", p, " & ")
end
print(file, "time elapsed lesioned PF & avg_V lesioned PF & mode realities lesioned PF \n")
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
#Perform particle filters

num_particles = 100

#num_samples to return
num_samples = 100

#num perturbation moves
num_moves = 1

#############################################
#Online model. Online MetaGen

#garbage, won't be used since lesion is false
V = Matrix{Float64}(undef, length(possible_objects), 2)

(traces, time_PF) = @timed particle_filter(num_particles, n_percepts, n_frames, gt_choices, num_samples, V, false);
print(file, "time elapsed particle filter  ", time_PF, " & ")

print(file, num_particles, " & ")
print(file, num_samples, " & ")
print(file, num_moves, " & ")

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects)

#############################################
#lesion with learned V. Retrospective MetaGen
avg_V = zeros(length(possible_objects), 2)
for i = 1:num_samples
	_,V,_ = Gen.get_retval(traces[i])
	global avg_V = avg_V + V/num_samples
	# println("R is ", R)
	# println("V is ", V)
end
V = avg_V

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

print_Vs_and_Rs_to_file(traces, num_samples, possible_objects, true)

close(file)
