#The file is for making inference of multiple percepts with one visual system. Must be run with an agrument for naming the output.txt file.

using Gen
using Distributions
using FreqTables
using Distances
using TimerOutputs

##############################################################################################
#Setting up helper functions

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


#small bug where all of the inputs need to by Float64. Doesn't accept Int64s
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

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ["BG", "person", "bicycle", "car", "motorcycle", "airplane",
               "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird",
               "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
               "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
               "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "baseball glove", "skateboard",
               "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
               "donut", "cake", "chair", "couch", "potted plant", "bed",
               "dining table", "toilet", "tv", "laptop","mouse", "remote",
               "keyboard", "cell phone", "microwave", "oven", "toaster",
               "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

#This function converts a list of category names to a list of category IDs. Specific to the COCO
#categories. Must have access to class_names.
function names_to_IDs(names::Vector{String}, possible_objects::Vector{String})
	IDs = Vector{Int}(undef, length(names))
	for i=1:length(names)
		#should only be one location of a given object
		IDs[i] = findfirst(isequal(names[i]),possible_objects)
	end
	return IDs
end

#This function converts a list of category names to an array of booleans which indicate whether the 
#object was present or not
function names_to_boolean(names::Vector{String}, possible_objects::Vector{String})
	booleans = zeros(length(possible_objects))
	for i=1:length(possible_objects)
		#should only be one location of a given object
		if possible_objects[i] in names
			booleans[i] = 1
		end
	end
	return booleans
end

##############################################################################################

@gen function sample_wo_repl(A,n)
	#now A itself should never change
	A_mutable = copy(A)
	A_immutable = copy(A)

	#println("A_immutable is ", A_immutable)
	#println("A_mutable is ", A_mutable)
	#println("n is ", n)

    sample = Array{String}(undef,n)
    for i in 1:n
    	#println("i is ", i)
    	
    	idx = @trace(Gen.uniform_discrete(1, length(A_mutable)), (:idx, i))
    	#print("idx is ", idx)
        #sample[i] = splice!(A_mutable, idx)
        sample[i] = A_mutable[idx]
        deleteat!(A_mutable, idx)
        #println("A_mutable is ", A_mutable)
    end
    #trying to reset A
    #A = copy(A_immutable)
    #want to rearrange so that the order of items in the sample matches the order of items that we're sampling from
    sampleIdx = names_to_IDs(sample, A_immutable)
    sorted = sort(sampleIdx)
    ordered_sample = A_immutable[sorted]
    return ordered_sample
end


#########

function metropolis_hastings2(trace, selection::Selection)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    (new_trace, weight) = regenerate(trace, args, argdiffs, selection)
    r,v,p = Gen.get_retval(new_trace)
    println(r)
    println(weight)
    if log(rand()) < weight
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end


#########


###################################################################################################################

alpha = 2
beta = 10

#Define generative model gm. gm takes as input the possible objects, the number of percepts to produce, and the number of frames
#per percepts.
@gen function gm(possible_objects::Vector{String}, n_percepts::Int, n_frames::Int)

	#need to make one possible_objects to change when replaced, another to not change?
	possible_objects_immutable = copy(possible_objects)

	#Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

	for j = 1:length(possible_objects_immutable)
        #set false alarm rate
        V[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to false alarm rate of around 0.1
        #set miss rate
        V[j,2] = @trace(Gen.beta(alpha, beta), (:m, j)) #leads to miss rate of around 0.1
	end

	#Determining frame of reality R
	lambda = 1 #must be <= length of possible_objects
	low = 0  #seems that low is never sampled, so this distribution will go from low+1 to high
	high = length(possible_objects_immutable)

	#generate each percept

    #percepts will contain many percepts. 
    percepts = []

    #Rs will contain many realities
    Rs = []

    for p = 1:n_percepts

        possible_objects_mutable = copy(possible_objects)

    	numObjects = @trace(trunc_poisson(lambda, low, high), (:numObjects, p))

        
        R = @trace(sample_wo_repl(possible_objects_mutable,numObjects), (:R, p))
        push!(Rs, R)


    	#Determing the percept based on the visual system V and the reality frame R
        #A percept is a matrix where each row is the percept for a frame.
    	percept = Matrix{Bool}(undef, n_frames, length(possible_objects_immutable))
    	for f = 1:n_frames
    		for j = 1:length(possible_objects_immutable)
    			#if the object is in the reality R, it is detected according to 1 - its miss rate
    			if possible_objects_immutable[j] in R
    				M =  V[j,2]
    				percept[f,j] = @trace(bernoulli(1-M), (:percept, p, f, j))
    			else
    				FA =  V[j,1]
    				percept[f,j] = @trace(bernoulli(FA), (:percept, p, f, j))
    			end
    		end
    	end

        push!(percepts, percept)

    end


	return (Rs,V,percepts); #returning reality R, (optional)
end;

###################################################################################################################

#Substituting for frequency table
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


#Analysis function needs the realities and Vs resulting from a sampling procedure and gt_V
function analyze(realities, Vs, gt_V, gt_R)

	#want to make a frequency table of the realities sampled
	ft = freqtable(realities)
	println(ft)
	dictionary = countmemb(realities)
	print(file ,dictionary, " & ")


	#compare means of Vs to gt_V
	#for false alarms
	##euclidean(gt_V[1], mean(Vs)[1])
	#for hit rates
	##euclidean(gt_V[2], mean(Vs)[2])




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
			count = get(dictionary,string(unique_realities[j]),0)
			avg_R = avg_R + names_to_boolean(unique_realities[j][p],possible_objects)*count/length(realities)
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
		distances = euclidean(avg_Rs[p], gt_R_bool)
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

	
	# #compare randomly generated V to groudtruth Vs
	# num_Rand_Vs = 10
	# (nrow,_) = size(gt_V)
	# avg_dist_FA = 0
	# avg_dist_M = 0
	# b = Distributions.Beta(2,10)
	# for idx=1:num_Rand_Vs
	# 	randVs = Matrix{Float64}(undef, nrow, 2)
	# 	for i = 1:nrow
 #    	    #set false alarm rate
 #    	    randVs[i,1] = rand(b,1) #leads to false alarm rate of around 0.1
 #    	    #set miss rate
 #    	    randVs[i,2] = rand(b,1) #leads to miss rate of around 0.1
	# 	end
	# 	avg_dist_FA = avg_dist_FA + (euclidean(gt_V[1], randVs[idx][1])/num_Rand_Vs)
	# 	avg_dist_M = avg_dist_M + (euclidean(gt_V[2], randVs[idx][2])/num_Rand_Vs)
	# end

end;



##################################################################################################################
#Helper functions for Metroplis Hastings

# Perform a single block resimulation update of a trace. method is a string specifying whether this is using the 
#metropolis_hastings
function block_resimulation_update(trace)
    
    (possible_objects, n_percepts, n_frames) = get_args(trace)
    n = length(possible_objects)

    #selection will keep track of things to select
    selection = select()

    #adding visual system parameters to selection
    for i = 1:n
    	push!(selection, (:fa,i))
        push!(selection, (:m,i))
    end

    for p = 1:n_percepts
        #adding reality to selection
        push!(selection, (:R, p))
        #adding numObjects to selection so the chain isn't stuck sampling the same numObjects each time
        push!(selection, (:numObjects, p))
    end

    r,v,per = Gen.get_retval(trace)
    println("current state is ",r)

    (trace, M) = metropolis_hastings2(trace, selection)

    trace
end;

function block_resimulation_inference((possible_objects, n_frames), observations)
    (tr, _) = generate(gm, (possible_objects, n_frames), observations)
    for iter=1:amount_of_computation_per_resample
        tr = block_resimulation_update(tr)
    end
    tr
end;

#One chain, look at every step of it
function every_step(possible_objects, n_percepts, n_frames, observations, amount_of_computation_per_resample)
	traces = []
	(tr, _) = generate(gm, (possible_objects, n_percepts, n_frames), observations) #starting point
	push!(traces,tr)
	for i=1:amount_of_computation_per_resample
    	tr = block_resimulation_update(tr)
    	push!(traces,tr)
	end
	traces
end;


###################################################################################################################
#Particle filter helper functions

#std controls the standard deviation of the normal perpurbations of the fa and miss rates
@gen function perturbation_proposal(prev_trace, std::Int)
    choices = get_choices(prev_trace)
    (T,) = get_args(prev_trace)
    #preturb fa and miss rates normally with std 0.1 May have to adjust so I don't get probabilities greater thatn 1 or less than 0
    for j = 1:length(possible_objects)
    	#new FA rate will be between 0 and 1
    	FA = @trace(trunc_normal(choices[(:fa, j)], std, 0.0, 1.0), (:fa, j))
        M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 1.0), (:m, j))
    end
end

# If I allowed a resample of V, that would defeat the purpose of posterior becoming new prior.
# Instead, just add some noise.
function perturbation_move(trace)
    Gen.metropolis_hastings(trace, perturbation_proposal, (0.1,))
end;


function particle_filter(num_particles::Int, gt_percept, num_samples::Int)

	# construct initial observations
	init_obs = Gen.choicemap()
	nrows,ncols = size(gt_percept[1])
	n_percepts = size(gt_percept)[1]

	println(n_percepts)

	p=1
	for i = 1:nrows
		for j = 1:ncols
			init_obs[(:percept,p,i,j)] = gt_percept[p][i,j]
		end
	end

	#initial state
	#num_percepts is 1 because starting off with just one percept
	state = Gen.initialize_particle_filter(gm, (possible_objects, 1, n_frames), init_obs, num_particles)


	for p = 2:n_percepts

		#tr = Gen.sample_unweighted_traces(state, num_samples)
		tr = get_traces(state)
		log_weights = get_log_weights(state)
		log_ml_estimate = Gen.log_ml_estimate(state)
		println("log_ml_estimate is ", log_ml_estimate)
		for i = 1:num_samples
			#trying to understand what's in state
			R,V,_ = Gen.get_retval(tr[i])
			println("initial R is ", R)
			println("initial V is ", V)
			println("log_weight is ", log_weights[i])
		end

		# apply rejuvenation/perturbation move to each particle. optional.
        for i = 1:num_particles
            state.traces[i], _ = perturbation_move(state.traces[i])
        end

		do_resample = Gen.maybe_resample!(state, ess_threshold=num_particles/2)
		println("do_resample ", do_resample)

		obs = Gen.choicemap()
		for i = 1:nrows
			for j = 1:ncols
				obs[(:percept,p,i,j)] = gt_percept[p][i,j]
			end
		end

		Gen.particle_filter_step!(state, (possible_objects, p, n_frames), (UnknownChange(),), obs)

	end
	# return a sample of unweighted traces from the weighted collection
	return Gen.sample_unweighted_traces(state, num_samples)
end;

###################################################################################################################

#creating output file
#outfile = string("output", ARGS[1], ".csv")
outfile = string("output1.csv")
file = open(outfile, "w")

#Defining observations / constraints
possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#each V sill have n_percepts, that many movies
n_percepts = 2 #particle filter is set up such that it needs at least 2 percepts
n_frames = 10


#file header
print(file, "gt_V & gt_R & ")
for p=1:n_percepts
	print(file, "p", p, " & ")
end
println(file, "Avg Euclidean distance FA between expectation and gt_V & Avg Euclidean distance M between expectation and ground truth V & time elapsed MH & amount_of_computation_per_resample & burnin & frequency table MH & unique_realities MH & mode realities MH & avg_Vs_binned for unique_realities MH & avg_Rs MH & Euclidean distance between avg_Rs and gt_R MH & Euclidean distance FA MH & Euclidean distance M MH & time elapsed PF & num_particles & num_samples & num_moves & frequency table PF & unique_realities PF & mode realities PF & avg_Vs_binned for unique_realities PF & avg_Rs PF & Euclidean distance between avg_Rs and gt_R PF & Euclidean distance FA PF & Euclidean distance M")


#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames))
gt_reality,gt_V,gt_percept = Gen.get_retval(gt_trace)	
print(file, gt_V, " & ")
print(file, "\"", gt_reality, "\" & ")

#Translating gt_percept back into names
for p = 1:n_percepts
	print(file, "\"")
	for f = 1:n_frames
		percept = []
		percept = possible_objects[gt_percept[p][f,:]]
		print(file,  percept)
	end
	print(file, "\" & ")
end


#get the percepts
#obs = Gen.get_submap(gt_choices, :percept
observations = Gen.choicemap()
for p = 1:n_percepts
	for i = 1:n_frames
		for j = 1:J
				observations[(:percept,p,i,j)] = gt_percept[p][i,j]
		end
	end
end


#distance between mean of the beta distribution and the ground truth Vs. Measurement of how unusual the V is.

(nrow,_) = size(gt_V)
b = Distributions.Beta(alpha,beta)
meanVs = ones(nrow,2) * mean(b)


print(file, euclidean(gt_V[1], meanVs[1]), " & ")
print(file, euclidean(gt_V[2], meanVs[2]), " & ")


#############################################
#Perform MH

#num_samples = 1
amount_of_computation_per_resample = 200 #????

(traces, time_MH) = @timed every_step(possible_objects, n_percepts, n_frames, observations, amount_of_computation_per_resample)
print(file, time_MH, " & ")

burnin = 100 #how many samples to ditch

print(file, amount_of_computation_per_resample, " & ")
print(file, burnin, " & ")

realities = Array{Array{String}}[]
Vs = Array{Float64}[]
for i = burnin + 1:amount_of_computation_per_resample
	Rs,V,_ = Gen.get_retval(traces[i])
	push!(Vs,V)
    push!(realities,Rs)
end


analyze(realities, Vs, gt_V, gt_reality)
print(file, " & ")

##############################################

#Perform particle filter

num_particles = 100

#num_samples to return
num_samples = 100

#num perturbation moves
num_moves = 1

#(traces,time_particle) = @timed particle_filter(num_particles, gt_percept, num_samples);
#println(file, "time elaped \n ", time_particle)
(traces, time_PF) = @timed particle_filter(num_particles, gt_percept, num_samples);
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
