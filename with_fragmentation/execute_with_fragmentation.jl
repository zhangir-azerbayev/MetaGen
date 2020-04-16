#Main script for executing data simulation and inference using the generative model
include("gm_with_fragmentation.jl")
include("inference_with_fragmentation.jl")

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs
using Random

#creating output file
#outfile = string("output", ARGS[1], ".csv")
outfile = string("output111.csv")
file = open(outfile, "w")

#Defining observations / constraints
possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#each V sill have n_percepts, that many movies
n_percepts = 3 #particle filter is set up such that it needs at least 2 percepts
n_frames = 1


#file header
print(file, "gt_V & gt_R & ")
for p=1:n_percepts
	print(file, "percept", p, " & ")
end
for p=1:n_percepts
	print(file, "avg V after p", p, " & ")
end
println(file, "time elapsed PF & num_particles & num_samples & num_moves & frequency table PF & unique_realities PF & mode realities PF & avg_Vs_binned for unique_realities PF & avg_Rs PF & Euclidean distance between avg_Rs and gt_R PF & Euclidean distance FA PF & Euclidean distance M PF &")

##########################

##For Simulated data

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames))
gt_reality,gt_V,gt_percepts = Gen.get_retval(gt_trace)
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

#Saving gt_percepts to file
percepts = []
for p = 1:n_percepts
	percept = []
	println("gt_percepts[p] ", gt_percepts[p])
	for f = 1:n_frames
		println("gt_percepts[p][f] ", gt_percepts[p][f])
		#println("possible_objects[gt_percepts[p][f]] ", possible_objects[gt_percepts[p][f]])
		perceived_frame = gt_percepts[p][f]
		print(file,  perceived_frame)
	end
	print(file, " & ")
end

println("percepts ", gt_percepts)


#store the visual counts in the submap to observations
#not sure this is even used
observations = Gen.choicemap()
for p = 1:n_percepts
	for f = 1:n_frames
		#observations[(:perceived_frame,p,f)] = gt_percepts[p][f]
		addr = (:perceived_frame,p,f) => :visual_count
		sm = Gen.get_submap(gt_choices, addr)
		Gen.set_submap!(observations, addr, sm)
	end
end

# ##############################################

#Perform particle filter

num_particles = 2 #100

#num_samples to return
num_samples = 2 #100

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

analyze(realities, Vs, gt_V, gt_reality)
#############################################


close(file)
