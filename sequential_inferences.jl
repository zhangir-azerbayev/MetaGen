using Gen
using Distributions
using FreqTables
using Distances


# function particle_filter_step2!(state::ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
#         observations::ChoiceMap) where {U}
#     num_particles = length(state.traces)
#     for i=1:num_particles
#         (state.new_traces[i], increment, _, discard) = update(
#             state.traces[i], new_args, argdiffs, observations)
#         if !isempty(discard)
#             error("Choices were updated or deleted inside particle filter step: $discard")
#         end
#         state.log_weights[i] += increment
#     end
    
#     # swap references
#     tmp = state.traces
#     state.traces = state.new_traces
#     state.new_traces = tmp
    
#     return nothing
# end


#Setting up helper functions

##############################################################################################

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
        sample[i] = splice!(A_mutable, idx)
        #sample[i] = A_mutable[idx]
        #deleteat!(A_mutable, idx)
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

#Define generative model gm. gm takes as input the possible objects, the number of percepts to produce, and the number of frames
#per percepts. n_percepts is kind of like time, since each percept, or video, is presented sequentially.
@gen function gm(possible_objects::Vector{String}, n_percepts::Int, n_frames::Int)

	#need to make one possible_objects to change when replaced, another to not change?
	possible_objects_immutable = copy(possible_objects)

	#Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

	for j = 1:length(possible_objects_immutable)
		#set false alarm rate
		V[j,1] = @trace(Gen.beta(2, 10), (:fa, j)) #leads to false alarm rate of around 0.1
		#set miss rate
		V[j,2] = @trace(Gen.beta(2, 10), (:m, j)) #leads to miss rate of around 0.1
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

##############################################################################################
#Defining observations / constraints

#Let me construct a test case to examine order effects. This should prove that the visual system carries  over.
#Version 1: learns high FA for bicycles, miss rate for airplanes. Then test on ambiguous
#versus
#Version 2: ambiguous, then later learning
#Should resolve ambiguous differently. Will it learn V differently? not sure.

possible_objects = ["person", "bicycle", "car","motorcycle", "airplane"]
n_frames = 10
n_percepts = 3

fake_percept1 = zeros(n_frames,length(possible_objects))
#ambiguous
fake_percept1[:,5] = [0,1,0,1,0,1,0,1,0,1] #airplane
fake_percept1[:,2] = [1,0,1,0,1,0,1,0,1,0] #bicycle

fake_percept2 = zeros(n_frames,length(possible_objects))
#clearly airplane, just misses sometimes
fake_percept2[:,5] = [1,0,1,1,1,1,1,0,1,1] #airplane missed twice
fake_percept2[:,2] = [1,0,0,0,0,0,0,1,0,0] #bicycle false alarmed twice

#learn V first. then try ambiguous case
fake_percepts = [fake_percept1, fake_percept2, fake_percept2]
#These are the sequential observations

####################################################



##############################################################################################
#Particle Filter

num_particles = 10000

#num_samples to return
num_samples = 10000

#num perturbation moves
num_moves = 1


##############################################################################################

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


function particle_filter(num_particles::Int, fake_percepts, num_samples::Int)

	# construct initial observations
	init_obs = Gen.choicemap()
	nrows,ncols = size(fake_percepts[1])
	n_percepts = size(fake_percepts)[1]

	p=1
	for i = 1:nrows
		for j = 1:ncols
			init_obs[(:percept,p,i,j)] = fake_percepts[p][i,j]
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

		# apply rejuvenation/perturbation moves to each particle. optional.
		for j = 1:num_moves
        	for i = 1:num_particles
            	state.traces[i], _ = perturbation_move(state.traces[i])
        	end
        end

		do_resample = Gen.maybe_resample!(state, ess_threshold=num_particles/2)
		println("do_resample ", do_resample)

		obs = Gen.choicemap()
		for i = 1:nrows
			for j = 1:ncols
				obs[(:percept,p,i,j)] = fake_percepts[p][i,j]
			end
		end

		Gen.particle_filter_step!(state, (possible_objects, p, n_frames), (UnknownChange(),), obs)

	end

	# return a sample of unweighted traces from the weighted collection
	return Gen.sample_unweighted_traces(state, num_samples)
end;


traces = particle_filter(num_particles, fake_percepts, num_samples);





###################################################################################################################

realities = Array{Array{String}}[]


Vs = Array{Float64}[]
for i = 1:num_samples
#for i = 1:length(traces)
	Rs,V,_ = Gen.get_retval(traces[i])
	push!(Vs,V)

    push!(realities,Rs)

end

###################################################################################################################
#Analysis

#want to make a frequency table of the realities sampled
ft = freqtable(realities)

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
#FA rates
avg_Vs_binned[idx][:,1];
#hit rates
avg_Vs_binned[idx][:,2];



#compare mean V of most frequent reality to gt_V
#for false alarms
##euclidean(gt_V[1], avg_Vs_binned[idx][1])
#for hit rates
##euclidean(gt_V[2], avg_Vs_binned[idx][2])

#compare to least frequent reatlity
#index of most frequent reality
##idx2 = findfirst(isequal(minimum(freq)),freq)
##unique_realities[idx2]

#compare mean V of most frequent reality to gt_V
#for false alarms
##euclidean(gt_V[1], avg_Vs_binned[idx2][1])
#for hit rates
##euclidean(gt_V[2], avg_Vs_binned[idx2][2]);
