#using PyCall

using Gen
using Distributions
using FreqTables
using Distances

#push!(pyimport("sys")["path"], pwd());
#pyimport("something.py")[:hello_world]()
#pythonFile = pyimport("something")

#pythonFile.hello_world()

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

function sample_momenta(n::Int)
    Float64[random(normal, 0, 1) for _=1:n]
end

function assess_momenta(momenta)
    logprob = 0.
    for val in momenta
        logprob += logpdf(normal, val, 0, 1)
    end
    logprob
end

#########

function hmc2(trace::U, selection::Selection;
             L=10, eps=0.1) where {T,U}
    prev_model_score = get_score(trace)
    args = get_args(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    argdiffs = map((_) -> NoChange(), args)

    # run leapfrog dynamics
    new_trace = trace
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    momenta = sample_momenta(length(values))
    prev_momenta_score = assess_momenta(momenta)
    for step=1:L

        # half step on momenta
        momenta += (eps / 2) * gradient

        # full step on positions
        values += eps * momenta

        # get new gradient
        values_trie = from_array(values_trie, values)
        (new_trace, _, _) = update(new_trace, args, argdiffs, values_trie)
        (_, _, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        gradient = to_array(gradient_trie, Float64)

        # half step on momenta
        momenta += (eps / 2) * gradient
    end

    # assess new model score (negative potential energy)
    new_model_score = get_score(new_trace)

    # assess new momenta score (negative kinetic energy)
    new_momenta_score = assess_momenta(-momenta)

    # accept or reject
    alpha = new_model_score - prev_model_score + new_momenta_score - prev_momenta_score
    if log(rand()) < alpha
        (new_trace, true)
    else
        (trace, false)
    end
end


#########

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
               "dining table", "toilet", "tv", "laptop", "mouse", "remote",
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

#Define generative model gm. gm takes as input the possible objects and the number of frames.
@gen function gm(possible_objects::Vector{String}, n_frames::Int)

	#need to make one possible_objects to change when replaced, another to not change?
	possible_objects_immutable = copy(possible_objects)
	possible_objects_mutable = copy(possible_objects)

	#Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

	for j = 1:length(possible_objects_immutable)
		#set false alarm rate
		V[j,1] = @trace(Gen.beta(1.909091, 36.272727), (:fa, j)) #leads to false alarm rate of 0.01
		#set miss rate
		V[j,2] = @trace(Gen.beta(1.909091, 36.272727), (:m, j)) #leads to miss rate of 0.05
	end

	#Determining frame of reality R
	lambda = 1 #must be <= length of possible_objects
	low = 0  #seems that low is never sampled, so this distribution will go from low+1 to high
	high = length(possible_objects_immutable)

	#println("possible_objects_mutable ", possible_objects_mutable)

	numObjects = @trace(trunc_poisson(lambda, low, high), :numObjects)
    R = @trace(sample_wo_repl(possible_objects_mutable,numObjects), :R) #order gets mixed up
    #why do changes to possible_objects_mutable last?

	#Determing the percept based on the visual system V and the reality frame R
    #A percept is a matrix where each row is the percept for a frame.
	percept = Matrix{Bool}(undef, n_frames, length(possible_objects_immutable))
	for f = 1:n_frames
		for j = 1:length(possible_objects_immutable)
			#if the object is in the reality R, it is detected according to 1 - its miss rate
			if possible_objects_immutable[j] in R
				M =  V[j,2]
				percept[f,j] = @trace(bernoulli(1-M), (:percept, f, j))
			else
				FA =  V[j,1]
				percept[f,j] = @trace(bernoulli(FA), (:percept, f, j))
			end
		end
	end


	return (R,V,percept); #returning reality R, (optional)
end;


##############################################################################################
#Defining observations / constraints

possible_objects = ["person", "bicycle", "car","motorcycle", "airplane"]
#Later, possible_objects will equal class_names

# define some observations
gt = Gen.choicemap()

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
n_frames = 10
gt_trace,_ = Gen.generate(gm, (possible_objects, n_frames))
gt_reality,gt_V,gt_percept = Gen.get_retval(gt_trace)

#println("gt_reality is ",gt_reality)
#println("gt_percept is ",gt_percept) #could translate back into names

# #Translating gt_percept back into names
# percept = Matrix{String}(undef, n_frames, length(possible_objects))
# for f = 1:n_frames
# 	percept[f,:] = possible_objects[gt_percept[f,:]]
# end
# println("percept is ",percept)

#gt_choices = Gen.get_choices(gt_trace)
#println(gt_choices)


# #get the percepts
# #obs = Gen.get_submap(gt_choices, :percept
# observations = Gen.choicemap()
# nrows,ncols = size(gt_percept)
# for i = 1:nrows
# 	for j = 1:ncols
# 			observations[(:percept,i,j)] = gt_percept[i,j]
# 	end
# end


fake_percept = zeros(n_frames,length(possible_objects))
#all person now
fake_percept[:,5] = [0,1,0,1,0,1,0,1,0,1] #airplane is fake
fake_percept[:,2] = [1,0,1,0,1,0,1,0,1,0] #bicycle is fake

# #for now, make the percepts
observations = Gen.choicemap()
nrows,ncols = size(fake_percept)
for i = 1:nrows
	for j = 1:ncols
			observations[(:percept,i,j)] = fake_percept[i,j]
	end
end

##################################################################################################################

#initialize a new trace
#trace, _ = Gen.generate(gm, (possible_objects, n_frames), observations)

#Inference procedure 1: Importance resampling


#add log_norm_weights as middle return arguement for importance_sampling
#(trace, lml_est) = Gen.importance_resampling(gm, (possible_objects, n_frames), observations, num_samples)
#(traces, log_norm_weights, lml_est) = Gen.importance_sampling(gm, (possible_objects, n_frames), observations, num_samples)
#for some reason importance_sampling seems not to account for observations

# #trying repeated importance_resampling
# amount_of_computation_per_resample = 10
# traces = []
# log_probs = Array{Float64}(undef, num_samples)
# for i = 1:num_samples
#      (tr, lml_est) = Gen.importance_resampling(gm, (possible_objects, n_frames), observations, amount_of_computation_per_resample)
#      push!(traces,tr)
#      log_probs[i] = Gen.get_score(tr)
# end


#################################################################################################################################

# #trying Metropolis Hastings
# num_samples = 100
# amount_of_computation_per_resample = 10 #????

# trace,_ = Gen.generate(gm, (possible_objects, n_frames), observations)

# # Perform a single block resimulation update of a trace.
# function block_resimulation_update(tr)

#     # Block 1: Update the reality
#     reality = select(:R)
#     (tr, _) = mh(tr, reality)
    
#     # Block 2: Update the visual system
#     (possible_objects, n_frames) = get_args(tr)
#     n = length(possible_objects)
#     for i = 1:n
#     	row_V = select((:(fa,i)),(:(m,i)))
#     	(tr, _) = mh(tr, row_V)
#     end
#     tr
# end;

# function block_resimulation_inference((possible_objects, n_frames), observations)
#     (tr, _) = generate(gm, (possible_objects, n_frames), observations)
#     for iter=1:amount_of_computation_per_resample
#         tr = block_resimulation_update(tr)
#     end
#     tr
# end;


# traces = []
# for i=1:num_samples
#     tr = block_resimulation_inference((possible_objects, n_frames), observations)
#     push!(traces,tr)
# end

#######################################################################################################################

#Metropolis Hastings all in one block

################
#Helper functions

# Perform a single block resimulation update of a trace. method is a string specifying whether this is using the 
#metropolis_hastings or hamiltonian method
function block_resimulation_update(trace, method)
    
    (possible_objects, n_frames) = get_args(trace)
    n = length(possible_objects)

    #selection will keep track of things to select
    selection = select()

    #adding visual system parameters to args
    for i = 1:n
    	push!(selection, (:fa,i))
        push!(selection, (:m,i))
    end

    #adding reality to selection
    push!(selection, :R)
    #adding numObjects to selection so the chain isn't stuck sampling the same numObjects each time
    push!(selection, :numObjects)

    r,v,p = Gen.get_retval(trace)
    println("current state is ",r)

    if isequal(method,"metropolis_hastings")
    	(trace, M) = metropolis_hastings2(trace, selection)
    elseif isequal(method,"hamiltonian")
    	(trace, _) = hmc2(trace, selection, L=10, eps=0.1)
    end

    trace
end;

function block_resimulation_inference((possible_objects, n_frames), observations)
    (tr, _) = generate(gm, (possible_objects, n_frames), observations)
    for iter=1:amount_of_computation_per_resample
        tr = block_resimulation_update(tr)
    end
    tr
end;


#################################################################################################################################


################

#Do sampling (works for MH or HMC)

num_samples = 1
amount_of_computation_per_resample = 10000 #????


# traces = []
# for i=1:num_samples
#     tr = block_resimulation_inference((possible_objects, n_frames), observations)
#     push!(traces,tr)
# end

#MH MCMC or HMC
#metropolis_hastings or hamiltonian
method = "metropolis_hastings"

#One chain, look at every step of it
function every_step(possible_objects, n_frames, observations)
	traces = []
	(tr, _) = generate(gm, (possible_objects, n_frames), observations) #starting point
	push!(traces,tr)
	for i=1:amount_of_computation_per_resample
    	tr = block_resimulation_update(tr,method)
    	push!(traces,tr)
	end
	traces
end;


traces = every_step(possible_objects, n_frames, observations)


#################################################################################################################################


burnin = 0 #how many samples to ditch

realities = Array{String}[]
Vs = Array{Float64}[]
for i = burnin+1:10000
#for i = 1:length(traces)
	reality,V,_ = Gen.get_retval(traces[i])
	push!(realities,reality)
	push!(Vs,V)
end

###################################################################################################################
#Analysis

#want to make a frequency table of the realities sampled
ft = freqtable(realities)

#compare means of Vs to gt_V
#for false alarms
euclidean(gt_V[1], mean(Vs)[1])
#for hit rates
euclidean(gt_V[2], mean(Vs)[2])


#want, for each reality, to bin Vs
unique_realities = unique(realities)
avg_Vs_binned = Array{Float64}[]
freq = Array{Float64}(undef, length(unique_realities))

for j = 1:length(unique_realities)
	index = findall(isequal(unique_realities[j]),realities)
	#freq keeps track of how many there are
	freq[j] = length(index)
	push!(avg_Vs_binned, mean(Vs[index]))
end

#find avg_Vs_binned at most common realities and compute euclidean distances
#index of most frequent reality
idx = findfirst(isequal(maximum(freq)),freq)
unique_realities[idx]

#compare mean V of most frequent reality to gt_V
#for false alarms
euclidean(gt_V[1], avg_Vs_binned[idx][1])
#for hit rates
euclidean(gt_V[2], avg_Vs_binned[idx][2])

#compare to least frequent reatlity
#index of most frequent reality
idx2 = findfirst(isequal(minimum(freq)),freq)
unique_realities[idx2]

#compare mean V of most frequent reality to gt_V
#for false alarms
euclidean(gt_V[1], avg_Vs_binned[idx2][1])
#for hit rates
euclidean(gt_V[2], avg_Vs_binned[idx2][2]);





#going to explode quickly as possible_objects gets larger and samples get larger



#want to compare Vs sampled to ground V

# for i=1:numIters
#     (tr, _) = Gen.importance_resampling(gm, (possible_objects, n_frames), obs, 2000)
#     putTrace!(viz, "t$(i)", serialize_trace(tr))
#     log_probs[i] = Gen.get_score(tr)
# end





# inference_history = Vector{typeof(trace)}(undef, N)
# for i = 1:N
# 	#selection = Gen.select(:fa, :m, :numObjects, :obj_selection)
# 	trace,_ = Gen.hmc(trace)
# 	inference_history[i] = trace
# end