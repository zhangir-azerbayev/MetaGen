#The file is for making inference of multiple percepts with one visual system. Must be run with an agrument for naming the output.txt file.
#includes location in the GM. In this GM, each percept (video) will be represented with a matrix padded with NaNs.


using Gen
using Distributions
using FreqTables
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

#This function should permute the three input vectors in the same way.
#Each vector must have the same length
@gen function mix_up(frame,xs,ys)
	frame_mixed = Vector{String}()
	xs_mixed = []
	ys_mixed = []


	#add assert statements about length

	n = length(frame)

	indices = collect(1:n)
	for i in 1:n
		ind = Gen.uniform_discrete(1, length(indices))

		index = indices[ind]

    	push!(frame_mixed, frame[index])
    	push!(xs_mixed, xs[index])
    	push!(ys_mixed, ys[index])

    	deleteat!(indices, ind)
	end
    return (frame_mixed, xs_mixed, ys_mixed)
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

#for recursively fragmenting
@gen function fragment(fragmentation_rate,countFrag,i)
	#countFrag will have the number of times already fragmented
	#i needs to increase every time fragment is called
	bern = @trace(Gen.bernoulli(fragmentation_rate),(:bern, i))
	i = i+1
	#if fragment,
	if bern
		countFrag = countFrag + 1
		countFrag = countFrag + fragment(fragmentation_rate, countFrag, i)
	end
	return countFrag
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
#x_size is how many pixels accross the image is
@gen function gm(possible_objects::Vector{String}, n_percepts::Int, n_frames::Int, x_size::Float64, y_size::Float64)

	#Determining visual system V. FA rate, Miss rate.
	V = Matrix{Float64}(undef, length(possible_objects), 2)

    #just for now, average number of hallucinations per category will be 0.2
    hallucination_lambda = 0.2
	#just for now, fragmentation lambda is 3
	fragmentation_lambda = 3.0 

	for j = 1:length(possible_objects)
        #set lambda for hallucination
        V[j,1] = @trace(Gen.poisson(hallucination_lambda), (:hall_lambda, j)) #leads to false alarm rate of around 0.1
        #set miss rate
        V[j,2] = @trace(Gen.beta(alpha, beta), (:m, j)) #leads to miss rate of around 0.1
	end

	#Determining movie of reality R
	lambda_objects = 2 #must be <= length of possible_objects
	low = 0.0  #seems that low is never sampled, so this distribution will go from low+1 to high
	high = length(possible_objects)

	#move is how much the object should be able to move from frame to frame
	move = 5.0 #should scale with x_size. How much it can move

	#generate each percept

    #percepts will contain many percepts. Here, each percept is a video
    percepts = []

    #will contain list of the coordinates of each object in each video
    gt_coordinates_x = []
    gt_coordinates_y = []

    #perceived. Will have to be an array so the length can be variable
    perceived_coordinates_x = []
    perceived_coordinates_y = []

    #Rs will contain many realities
    Rs = []

    for p = 1:n_percepts  #for each video

        possible_objects_mutable = copy(possible_objects)

    	numObjects = @trace(Gen.poisson(lambda_objects), (:numObjects, p))

        
        R = @trace(sample_with_repl(possible_objects_mutable,numObjects), (:R, p))
        println("R ", R)
        push!(Rs, R)

        #Each row is a frame and each column is an object
        #Ground truth
        gt_coordinate_x = Matrix{Float64}(undef, n_frames, numObjects)
        gt_coordinate_y = Matrix{Float64}(undef, n_frames, numObjects)


    	#Determing the percept based on the visual system V and the reality frame R
        #A percept is a matrix where each row is the percept for a frame and each column is a potential object.
        #Keeps track of the objects seen
        #n_columns will be big
        n_columns = 1000
    	percept = Matrix{String}(undef, n_frames, n_columns)
        percept_x = Matrix{Float64}(undef, n_frames, n_columns)
        percept_y = Matrix{Float64}(undef, n_frames, n_columns)
        fill!(percept, "empty")
        fill!(percept_x, NaN)
        fill!(percept_y, NaN)

    	for f = 1:n_frames

    		#first frame, initialize the coordinates
    		if f == 1
    			#for each object in reality, initialize it to a random location. 
    			for j = 1:numObjects
    				gt_coordinate_x[f,j] = @trace(Gen.uniform(0,x_size), (:gt_coordinate_x, p, f, j))
    				gt_coordinate_y[f,j] = @trace(Gen.uniform(0,y_size), (:gt_coordinate_y, p, f, j))
    			end
    		#later frames. Model of motion goes here
    		else
    			for j = 1:numObjects
    				#normal around previous location. Limits are low and x_size
    				gt_coordinate_x[f,j] = @trace(trunc_normal(gt_coordinate_x[f,j], move, low, x_size), (:gt_coordinate_x, p, f, j))
    				gt_coordinate_y[f,j] = @trace(trunc_normal(gt_coordinate_y[f,j], move, low, y_size), (:gt_coordinate_y, p, f, j))
    			end
    		end

    		#Then coordinates have to be perceived with some noise
    		noise = 1.0 #should scale with x_size
    		i = 0 #keeps track of indexing for coordinates
    		for r = 1:length(R)
                reality = R[r]
    			#get ID
    			j = names_to_IDs([reality], possible_objects)
    			#detected with probability according to 1 - its miss rate
    			M =  V[j,2][1]
    			#if detected
    			detected = @trace(Gen.bernoulli(1-M), (:detection, p, f, r))



                #If it will fragment, how many times?
                fragmentation_count = @trace(trunc_poisson(fragmentation_lambda,0.0,10.0), (:fragmententation_count, p, f, r))

                i = i + 1
                if detected
                    #add the object
                    percept[f,i] = R[r]
                    percept_x[f,i] = @trace(trunc_normal(gt_coordinate_x[f,r], noise, low, x_size), (:percept_x, p, f, i))
                    percept_y[f,i] = @trace(trunc_normal(gt_coordinate_y[f,r], noise, low, y_size), (:percept_y, p, f, i))

                    #add locations for each fragmentations
                    for frag=1:fragmentation_count
                        i = i + 1
                        #add the object to the frame
                        percept[f,i] = R[r]
                        percept_x[f,i] = @trace(trunc_normal(gt_coordinate_x[f,r], noise, low, x_size), (:percept_x, p, f, i))
                        percept_y[f,i] = @trace(trunc_normal(gt_coordinate_y[f,r], noise, low, y_size), (:percept_y, p, f, i))
                    end
                else
                    #did 999 instead of NaN

                    #don't add the object
                    #percept[f,i] = undef #unnecessary. should aready have NaNs or undef
                    percept_x[f,i] = @trace(Gen.bernoulli(1.0), (:percept_x, p, f, i))
                    percept_y[f,i] = @trace(Gen.bernoulli(1.0), (:percept_y, p, f, i))

                    #add locations for each fragmentation (that didn't really happen)
                    for frag=1:fragmentation_count
                        i = i + 1
                        #add the object to the frame
                        #percept[f,i] = undef #unnecessary. should aready have NaNs or undef
                        percept_x[f,i] = @trace(Gen.bernoulli(1.0), (:percept_x, p, f, i))
                        percept_y[f,i] = @trace(Gen.bernoulli(1.0), (:percept_y, p, f, i))
                    end
                end
    		end

    		#hallucinations
    		#for each possible object category, hall_lam
    		for j = 1:length(possible_objects)
                possible_object = possible_objects[j]
    			#j = names_to_IDs([possible_object], possible_objects)
    			hall_lam =  V[j,1][1]
    			hallucination_count = @trace(trunc_poisson(hall_lam,0.0,10.0), (:hallucination_count, p, f, j))

    			for hall=1:hallucination_count
    				i = i + 1
    				#add the object to the frame
    				percept[f,i] = possible_object
    				#random location
    				percept_x[f,i] = @trace(Gen.uniform(0,x_size), (:percept_x, p, f, i))
    				percept_y[f,i] = @trace(Gen.uniform(0,y_size), (:percept_y, p, f, i))
    			end
    		end

    		#need to mix up the order of everything in the frame so that the order isn't indicative of which observations
    		#are true and which aren't
    		#need to mix up frame, obs_coordinate_x, obs_coordinate_x.

            #alternatively, could sort by order in COCO dataset


            #println("percept ", percept)

            println("i ", i)
            #println("percept[f] ", percept[f, :])


            #should probably mix everything, not just the first i

            frame = percept[f, 1:i]
            frame_x = percept_x[f, 1:i]
            frame_y = percept_y[f, 1:i]

    		#println("frame ", frame)
            println("just before mix up")
    		(frame_mixed, frame_x_mixed, frame_y_mixed) = @trace(mix_up(frame, frame_x, frame_y), ((:frame_mixed, p, f), (:frame_x_mixed, p, f), (:frame_y_mixed, p, f)))
    		#println("frame_mixed ", frame_mixed)
            println("just after mix up")

    		percept[f, 1:i] = frame_mixed
    		percept_x[f, 1:i] = frame_x_mixed
    		percept_y[f, 1:i] = frame_y_mixed
    	end

        push!(percepts, percept)
        push!(gt_coordinates_x, gt_coordinate_x)
        push!(gt_coordinates_y, gt_coordinate_y)
        push!(perceived_coordinates_x, percept_x)
        push!(perceived_coordinates_y, percept_y)
    end

    println("Rs ", Rs)

	return (Rs,V,percepts, gt_coordinates_x, gt_coordinates_y, perceived_coordinates_x, perceived_coordinates_y); 
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
    	push!(selection, (:hall_lam,i))
        push!(selection, (:m,i))
    end

    for p = 1:n_percepts
        #adding reality to selection
        push!(selection, (:R, p))
        #adding numObjects to selection so the chain isn't stuck sampling the same numObjects each time
        push!(selection, (:numObjects, p))
        #
        #(:gt_coordinate_x, p, f, j) where j cycles through numObjects. Will likely cause problems. Might automatically do this
        #since gt_coordinate_x is downstream of R?
        # for j = 1:numObjects
        #     push!(selection, (:gt_coordinate_x, p, f, j))
        #     push!(selection, (:gt_coordinate_y, p, f, j))
        # end
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
	(tr, _) = generate(gm, (possible_objects, n_percepts, n_frames, x_size, y_size), observations) #starting point
	push!(traces,tr)
	for i=1:amount_of_computation_per_resample
    	tr = block_resimulation_update(tr)
    	push!(traces,tr)
	end
	traces
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

#perturbs V all at once. after more percepts, MH starts to reject the proposals a lot
#std controls the standard deviation of the normal perpurbations of the fa and miss rates
@gen function perturbation_proposal(prev_trace, std::Float64)
    choices = get_choices(prev_trace)
    #(T,) = get_args(prev_trace)
    #perturb fa and miss rates normally with std 0.1 May have to adjust so I don't get probabilities greater thatn 1 or less than 0
    for j = 1:length(possible_objects)
    	#new FA rate will be between 0 and 1
    	FA = @trace(trunc_normal(choices[(:hall_lam, j)], std, 0.0, 1.0), (:hall_lam, j))
        M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 1.0), (:m, j))
    end
end

#perturb each entry of V independently
#j is the index of the possible object whose FA or M will be perturbed
#FA is a boolean for if it will perturb the FA or M. If true, perturb FA. If false, perturb M.
#std controls the standard deviation of the normal perpurbations of the fa and miss rates
@gen function perturbation_proposal_individual(prev_trace, std::Float64, j::Int, FA::Bool)
    choices = get_choices(prev_trace)
    if FA
		#new FA rate will be between 0 and 1
    	FA = @trace(trunc_normal(choices[(:hall_lam, j)], std, 0.0, 1.0), (:hall_lam, j))
    else
    	#new M rate will be between 0 and 1
        M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 1.0), (:m, j))
    end



    # for reality in R
    # 			r = r + 1
    # 			#get ID
    # 			j = names_to_IDs([reality], possible_objects)
    # 			#detected with probability according to 1 - its miss rate
    # 			M =  V[j,2][1]
    # 			#if detected
    # 			if @trace(Gen.bernoulli(1-M), (:detection, p, f, r))
    # 				i = i + 1

    # 				#add the object to the frame
    # 				push!(frame,reality)

    # 				#perceive location with slight noise
    # 				println("here")
    # 				obs_coordinate_x = @trace(trunc_normal(gt_coordinate_x[f,r], noise, low, x_size), (:obs_coordinate_x, p, f, i))
    # 				obs_coordinate_y = @trace(trunc_normal(gt_coordinate_y[f,r], noise, low, y_size), (:obs_coordinate_y, p, f, i))
    # 				#Indexing got out of hand here. It goes, p, f, r, frag

    # 				push!(frame_x, obs_coordinate_x)
    # 				push!(frame_y, obs_coordinate_y)


    # 				#how many times does it fragment?
    # 				#1 for detection plus however many fragmentations happen
    # 				fragmentation_count = @trace(fragment(fragmentation_rate,0,1), (:fragmententation_count, p, f, r))

    # 				#add locations for each fragmentations
    # 				for frag=1:fragmentation_count
    # 					i = i + 1

    # 					#add the object to the frame
    # 					push!(frame,reality)
    # 					obs_coordinate_x = @trace(trunc_normal(gt_coordinate_x[f,r], noise, low, x_size), (:obs_coordinate_x, p, f, i))
    # 					obs_coordinate_y = @trace(trunc_normal(gt_coordinate_y[f,r], noise, low, y_size), (:obs_coordinate_y, p, f, i))

    # 					push!(frame_x, obs_coordinate_x)
    # 					push!(frame_y, obs_coordinate_y)
    # 				end
    # 			end
    # 		end


end

# If I allowed a resample of V, that would defeat the purpose of posterior becoming new prior.
# Instead, just add some noise.
function perturbation_move(trace)

	#Choose order of perturbation proposals randomly
	#mix up the order of the permutations
	#2 * for FA and M
    mixed_up = collect(1:2*length(possible_objects))
    mixed_up = Random.shuffle!(mixed_up)
    println("mixed up ", mixed_up)
	for j = 1:length(mixed_up)
		i = mixed_up[j]
		index = floor((i+1)/2)
		println("index ",index)
        convert(Int, index)
		trace, _ = Gen.metropolis_hastings(trace, perturbation_proposal_individual, (0.1,index,isodd(i)))
	end
	return trace
end;



function particle_filter(num_particles::Int, gt_percept, perceived_coordinates_x, perceived_coordinates_y, num_samples::Int)

	# construct initial observations
	init_obs = Gen.choicemap()
	n_percepts = size(gt_percept)[1]
	n_frames = size(gt_percept[1])[1]

	println(n_percepts)

	p=1
	for f = 1:n_frames
		init_obs[(:frame_mixed, p, f)] = gt_percept[p][f, :]
		init_obs[(:frame_x_mixed, p, f)] = perceived_coordinates_x[p][f,:]
		init_obs[(:frame_y_mixed, p, f)] = perceived_coordinates_y[p][f,:]
	end

	#initial state
	#num_percepts is 1 because starting off with just one percept
	state = Gen.initialize_particle_filter(gm, (possible_objects, 1, n_frames, x_size, y_size), init_obs, num_particles)


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
			obs[(:frame_mixed, p, f)] = gt_percept[p][f,:]
			obs[(:frame_x_mixed, p, f)] = perceived_coordinates_x[p][f,:]
			obs[(:frame_y_mixed, p, f)] = perceived_coordinates_y[p][f,:]
		end

		Gen.particle_filter_step!(state, (possible_objects, p, n_frames, x_size, y_size), (UnknownChange(),), obs)

		(log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    	ess = effective_sample_size(log_normalized_weights)
        println("ess after particle filter step is ", ess)

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
#J = length(possible_objects)

#each V sill have n_percepts, that many movies
n_percepts = 2 #particle filter is set up such that it needs at least 2 percepts
n_frames = 10

#for locations stuff
x_size = 10.0
y_size = 10.0


#file header
print(file, "gt_V & gt_R & ")
for p=1:n_percepts
	print(file, "percept", p, " & ")
end
print(file, "Avg Euclidean distance FA between expectation and gt_V & Avg Euclidean distance M between expectation and gt_V & time elapsed MH & amount_of_computation_per_resample & burnin & frequency table MH & unique_realities MH & mode realities MH & avg_Vs_binned for unique_realities MH & avg_Rs MH & Euclidean distance between avg_Rs and gt_R MH & Euclidean distance FA MH & Euclidean distance M MH & ")
for p=1:n_percepts
	print(file, "avg V after p", p, " & ")
end
println(file, "time elapsed PF & num_particles & num_samples & num_moves & frequency table PF & unique_realities PF & mode realities PF & avg_Vs_binned for unique_realities PF & avg_Rs PF & Euclidean distance between avg_Rs and gt_R PF & Euclidean distance FA PF & Euclidean distance M PF &")


#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames, x_size, y_size))
gt_reality, gt_V, gt_percept, gt_coordinates_x,gt_coordinates_y, perceived_coordinates_x, perceived_coordinates_y = Gen.get_retval(gt_trace)

println("gt_coordinates_x ", gt_coordinates_x)
println("gt_coordinates_y ", gt_coordinates_y)

println("perceived_coordinates_x ", gt_coordinates_x)
println("perceived_coordinates_y ", gt_coordinates_y)


gt_R_bool = names_to_boolean
print(file, gt_V, " & ")
print(file, gt_reality, " & ")

#Print the percepts to the file
for p = 1:n_percepts
	for f = 1:n_frames
		#percept = []
        #filter(elem -> isdefined(gt_percept[p][f,:], elem), 1:length(gt_percept[p][f,:]))
		percept = gt_percept[p][f,:]
		print(file,  percept)
	end
	print(file, " & ")
end


#get the percepts
#obs = Gen.get_submap(gt_choices, :percept
#add the coordinates
observations = Gen.choicemap()
for p = 1:n_percepts
	for f = 1:n_frames
		observations[(:frame_mixed, p, f)] = gt_percept[p][f,:]
		observations[(:frame_x_mixed, p, f)] = perceived_coordinates_x[p][f,:]
		observations[(:frame_y_mixed, p, f)] = perceived_coordinates_y[p][f,:]
	end
end

println("made it!!!")

#distance between mean of the beta distribution and the ground truth Vs. Measurement of how unusual the V is.

(nrow,_) = size(gt_V)
b = Distributions.Beta(alpha,beta)
meanVs = ones(nrow,2) * mean(b)


print(file, euclidean(gt_V[1], meanVs[1]), " & ")
print(file, euclidean(gt_V[2], meanVs[2]), " & ")


# #############################################
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

# ##############################################

#Perform particle filter

num_particles = 10

#num_samples to return
num_samples = 10

#num perturbation moves
num_moves = 1

#(traces,time_particle) = @timed particle_filter(num_particles, gt_percept, num_samples);
#println(file, "time elaped \n ", time_particle)
(traces, time_PF) = @timed particle_filter(num_particles, gt_percept, perceived_coordinates_x, perceived_coordinates_y, num_samples);
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
