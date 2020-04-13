#The file is for making inference of multiple percepts with one visual system. Must be run with an argument for naming the output.txt file.
#This file is identical to particle_filters_only.jl
#This file accomodates fragmentation.

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs
using Random

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
class_names = ["person", "bicycle", "car", "motorcycle", "airplane",
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

##############################################################################################

@gen function sample_with_repl(A,n)
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

##############################################################################################

#This function builds the percept for a frame. As input, it takes the reality R,the visual system V,
# fragmentation_lambda, fragmentation_max, hallucination_max, and possible_objects
#fragmentation_max will be the most fragmentations possible per object. So for a single token object in reality,
#it can be fragmented at most fragmentation_max times, in addition to being detected once.
@gen function build_percept(R, V::Matrix{Float64}, fragmentation_lambda::Float64, fragmentation_max::Float64, hallucination_max::Float64, possible_objects)
	hard_cap = 20.0 #hard cap on number of times an object can be perceived
	perceived_frame = []

	for j = 1:length(possible_objects)
		possible_object = possible_objects[j]

		#hallucinations
		hall_lam =  V[j,1][1]
		#since it won't sample 0.0 if 0.0 is low, setting low to -1.0
		#hallucination_count = @trace(trunc_poisson(hall_lam,-1.0,hallucination_max), (:hallucination_count, j))
		hallucination_count = @trace(trunc_poisson(hall_lam,-1.0,hallucination_max), (:hallucination_count => j))
		hallucination_count = convert(Float64, hallucination_count)

		#detections and fragmentations
		if possible_object in R
			#miss rate
			M =  V[j,2][1]
			#how many times detected?

			#can be 0, 1, ... , fragmentation_max + 1 times
			n = fragmentation_max+2
			n = convert(Int64, n)
			prob = Array{Float64}(undef, n)
			#probability of seeing nothing
			prob[1] = M
			for i = 2:n
				#how many fragmentations there are
				x = i-2
				#TODO might need low to be -1.0
				prob[i] = exp(Gen.logpdf(trunc_poisson, x, fragmentation_lambda, 0.0, fragmentation_max))
			end
			prob[2:n] = (1-M)*prob[2:n]/sum(prob[2:n])


			(M < 0.001) && println("M ", prob[1])

			#categorical returns an int between 1 and length(probs). I want to adjust the index a little so a miss is 0
			detection_count = @trace(categorical(prob), :detection_count  => j)
			detection_count = detection_count-1
			detection_count = convert(Float64, detection_count)

			visual_count = @trace(trunc_poisson(detection_count + hallucination_count, -1.0, hard_cap), (:visual_count => j))

		else #if object isn't in reality, visual_count only depends on hallucination_count
			visual_count = @trace(trunc_poisson(hallucination_count, -1.0, hard_cap), (:visual_count => j))
		end #end if

		for i = 1:visual_count
			push!(perceived_frame, possible_object)
		end

	end #end for
	return perceived_frame
end


###################################################################################################################

alpha = 2
beta = 10

#Define generative model gm. gm takes as input the possible objects, the number of percepts to produce, and the number of frames
#per percepts.
@gen function gm(possible_objects::Vector{String}, n_percepts::Int, n_frames::Int)

	#need to make one possible_objects to change when replaced, another to not change?
	possible_objects_immutable = copy(possible_objects)

	#just for now, fragmentation lambda is 3
	fragmentation_lambda = 3.0
	#just for now, average number of hallucinations per category will be 0.2
    mean_hallucination_lambda = 0.2
	std_hallucination_lambda = 0.5

    #max fragmentation
    fragmentation_max = 10.0
    hallucination_max = 10.0

	#Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

	for j = 1:length(possible_objects)
        #set lambda for hallucination. The lambda parameter is sampled from
		#a truncated normal distribution with mean hallucination_lambda, minimum
		#0, and maximum 100.0, which is completely arbitrary. The STD I set to
		#2, which is pretty arbitrary
        V[j,1] = @trace(trunc_normal(mean_hallucination_lambda, std_hallucination_lambda, 0.0,  100.0), (:hall_lambda, j)) #E[hall_lambda] is 0.2
        #set miss rate
        V[j,2] = @trace(Gen.beta(alpha, beta), (:m, j)) #leads to miss rate of around 0.1
	end

	#Determining frame of reality R
	lambda_objects = 2 #must be <= length of possible_objects
	low = 0  #seems that low is never sampled, so this distribution will go from low+1 to high
	high = length(possible_objects_immutable)

	#generate each percept

    #percepts will contain many percepts.
    percepts = []

    #Rs will contain many realities
    Rs = []

    for p = 1:n_percepts

        possible_objects_mutable = copy(possible_objects)

    	numObjects = @trace(Gen.poisson(lambda_objects), (:numObjects, p))


        R = @trace(sample_with_repl(possible_objects_mutable,numObjects), (:R, p))
        push!(Rs, R)

    	percept = []
    	for f = 1:n_frames
    		perceived_frame = @trace(build_percept(R, V, fragmentation_lambda, fragmentation_max, hallucination_max, possible_objects), (:perceived_frame, p, f))
    		push!(percept, perceived_frame)
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
			total = get(dictionary,string(unique_realities[j]),0)
			avg_R = avg_R + names_to_boolean(unique_realities[j][p],possible_objects)*total/length(realities)
		end
		push!(avg_Rs, avg_R)
	end

	# boolean_Rs = []
	# for p = 1:n_percepts
	# 	for j = 1:length(realities)
	# 		boolean_R = names_to_boolean(realities[j][p],possible_objects)
	# 		push!(boolean_Rs, boolean_R)
	# 	end
	# end
	# sum(boolean_Rs)


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


###################################################################################################################
#Particle filter helper functions

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

# #perturbs V all at once. after more percepts, MH starts to reject the proposals a lot
# #std controls the standard deviation of the normal perpurbations of the fa and miss rates
# @gen function perturbation_proposal(prev_trace, std::Float64)
#     choices = get_choices(prev_trace)
#     #(T,) = get_args(prev_trace)
#     #perturb fa and miss rates normally with std 0.1

#     for j = 1:length(possible_objects)
#     	#new FA rate will be between 0 and 1
#     	FA = @trace(trunc_normal(choices[(:fa, i)], std, 0.0, 1.0), (:fa, i))
#         M = @trace(trunc_normal(choices[(:m, i)], std, 0.0, 1.0), (:m, i))
#     end
# end

#perturb each entry of V independently
#j is the index of the possible object whose hall_lambda or M will be perturbed
#hall is a boolean for if it will perturb the hall_lambda or M. If true, perturb FA. If false, perturb M.
#std controls the standard deviation of the normal perpurbations of the fa and miss rates
@gen function perturbation_proposal_individual(prev_trace, std::Float64, j::Int, hall::Bool)
    choices = get_choices(prev_trace)
    if hall
		#new hallucination_lamda will be drawn from a normal distribution
		#centered on previous hallucination_lambda and between 0 and something arbitrary.
		#to do this right, might not want the upper bound. upper_bound is arbitrary.
		#exact value shouldn't matter because should be very unlikely anyway
		upper_bound = 100.0
    	hall = @trace(trunc_normal(choices[(:hall_lambda, j)], std, 0.0, upper_bound), (:hall_lambda, j))
    else
    	#new M rate will be between 0 and 1
        M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 1.0), (:m, j))
    end
end

# If I allowed a resample of V, that would defeat the purpose of posterior becoming new prior.
# Instead, just add some noise.
function perturbation_move(trace)

	# #Choose order of perturbation proposals randomly
	# #mix up the order of the permutations
	# #2 * for FA and M
    # mixed_up = collect(1:2*length(possible_objects))
    # mixed_up = Random.shuffle!(mixed_up)
	# for j = 1:length(mixed_up)
	# 	i = mixed_up[j]
	# 	index = floor((i+1)/2)
	# 	trace, _ = Gen.metropolis_hastings(trace, perturbation_proposal_individual, (0.1,index,isodd(i)))
	# end
	return trace
end;


function particle_filter(num_particles::Int, gt_percepts, gt_choices, num_samples::Int)

	n_percepts = size(gt_percepts)[1]
	n_frames = size(gt_percepts[1])[1]

	println("in particle_filter")

	# construct initial observations
	init_obs = Gen.choicemap()
	p=1
	for f = 1:n_frames
		#init_obs[(:perceived_frame,p,f)] = gt_percepts[p][f]
		addr = (:perceived_frame,p,f) => :visual_count
		sm = Gen.get_submap(gt_choices, addr)
		Gen.set_submap!(init_obs, addr, sm)
	end

	println("init_obs ", init_obs)

	#initial state
	#num_percepts is 1 because starting off with just one percept
	state = Gen.initialize_particle_filter(gm, (possible_objects, 1, n_frames), init_obs, num_particles)

	for p = 2:n_percepts

		println("percept ", p-1)

		#tr = Gen.sample_unweighted_traces(state, num_samples)
		tr = get_traces(state)
		log_weights = get_log_weights(state)
		log_ml_estimate = Gen.log_ml_estimate(state)
		println("log_ml_estimate is ", log_ml_estimate)

    	(log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    	ess = effective_sample_size(log_normalized_weights)
    	println("ess at start of loop is ", ess)

		if isnan(ess)
			t = filter(t -> isinf(get_score(t)), state.traces)
			ts = map(Gen.get_choices, t)
			println("here here")
			show(stdout, "text/plain", ts)
		end




		# #how does mean V change?
		# #initialize something for tracking average V
		# avg_V = Matrix{Float64}(undef, length(possible_objects), 2)
		# for i = 1:num_samples
		# 	#trying to understand what's in state
		# 	R,V,_ = Gen.get_retval(tr[i])
		# 	println("initial R is ", R)
		# 	println("initial V is ", V)
		# 	println("log_weight is ", log_weights[i])

		# 	avg_V = avg_V + V/num_samples
		# end
		# println("avg_V ", avg_V)


		# return a sample of unweighted traces from the weighted collection
		tr = Gen.sample_unweighted_traces(state, num_samples)

		#initialize something for tracking average V
		avg_V = zeros(length(possible_objects), 2)
		for i = 1:num_samples
			R,V,_ = Gen.get_retval(tr[i])
			avg_V = avg_V + V/num_samples
			println("R is ", R)
			println("V is ", V)
		end
		println("avg_V is ", avg_V)
		print(file, avg_V, " & ")


		# apply rejuvenation/perturbation move to each particle. optional.
        for i = 1:num_particles
        	R,V,_ = Gen.get_retval(state.traces[i])
        	println("V before perturbation ", V)

            state.traces[i] = perturbation_move(state.traces[i])

            R,V,_ = Gen.get_retval(state.traces[i])
            #println("R after perturbation is ", R)
			println("V after perturbation ", V)
			#println("log_weight after perturbation is ", log_weights[i])
        end

        (log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    	ess = effective_sample_size(log_normalized_weights)
        println("ess after perturbation is ", ess)

		do_resample = Gen.maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)


        (log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    	ess = effective_sample_size(log_normalized_weights)
        println("ess after resample is ", ess)

		obs = Gen.choicemap()
		for f = 1:n_frames
			#obs[(:perceived_frame,p,f)] = gt_percepts[p][f]
			addr = (:perceived_frame,p,f) => :visual_count
			sm = Gen.get_submap(gt_choices, addr)
			Gen.set_submap!(obs, addr, sm)
		end

		Gen.particle_filter_step!(state, (possible_objects, p, n_frames), (UnknownChange(),), obs)

		(log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    	ess = effective_sample_size(log_normalized_weights)
        println("ess after particle filter step is ", ess)

		if isnan(ess)
			t = filter(t -> isinf(get_score(t)), state.traces)
			ts = map(Gen.get_choices, t)
			println("here here")
			show(stdout, "text/plain", ts)
		end
	end
	# return a sample of unweighted traces from the weighted collection
	tr = Gen.sample_unweighted_traces(state, num_samples)

	println("percept ", n_percepts)

	#initialize something for tracking average V
	avg_V = zeros(length(possible_objects), 2)
	for i = 1:num_samples
		R,V,_ = Gen.get_retval(tr[i])
		avg_V = avg_V + V/num_samples
		println("R is ", R)
		println("V is ", V)
	end
	print(file, avg_V, " & ")

	return tr
end;

###################################################################################################################

#creating output file
#outfile = string("output", ARGS[1], ".csv")
outfile = string("output111.csv")
file = open(outfile, "w")

#Defining observations / constraints
possible_objects = ["person","bicycle","car","motorcycle","airplane"]
J = length(possible_objects)

#each V sill have n_percepts, that many movies
n_percepts = 3 #particle filter is set up such that it needs at least 2 percepts
n_frames = 10


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
println("gt_choices ", gt_choices);

#TODO will have to find a way to generate gt_choices from gt_percepts
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
