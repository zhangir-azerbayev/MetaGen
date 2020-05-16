#Generative model with fragmentation file
using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs

include("shared_functions.jl")
include("mbrfs_simplified.jl")



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

	# println("A_immutable is ", A_immutable)
	# println("A_mutable is ", A_mutable)
	# println("n is ", n)

    sample = Array{String}(undef,n)
    for i in 1:n
    	#println("i is ", i)

    	idx = @trace(Gen.uniform_discrete(1, length(A_mutable)), (:idx, i))
    	#println("idx is ", idx)
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



##############################################################################################

#This function builds the percept for a frame. As input, it takes the reality R,the visual system V,
#and the possible_objects.
@gen function build_percept(R, V::Matrix{Float64}, locations, possible_objects)
	#C has two columns, for the two ways of seeing something (correct detection and hallucination)
	#Each row is for a possible_category.
	#C = Matrix{Union{Array{Array{Float64,1},1}, Array{Any,1}}}(undef, length(possible_objects), 2)
	#Array{Array{Float64,1},1} and Array{Any,1} are the two possible output types from brfs

    C = Array{Union{Array{Array{Float64,1},1}, Array{Any,1}}}(undef, length(possible_objects))

	for j = 1:length(possible_objects)
		possible_object = possible_objects[j]

		FA =  V[j,1][1]
		M =  V[j,2][1]
        #contains
		E = (possible_object in R)
        #epsilon = 0.0001
        #E = (possible_object in R) ? 1-epsilon : epsilon

		#params for mvnormal for location
		mu = Vector{Float64}(undef, 2)
		cov = Matrix{Float64}(undef, 2, 2)
		sd_x = 1
		sd_y = 1

        #This part is redundant now
		# mu[1] = E ? locations[j,1] : -1 #x-coordinate #else doesn't matter
		# mu[2] = E ? locations[j,2] : -1 #y-coordinate #else doesn't matter

        mu[1] = locations[j,1] #x-coordinate #else doesn't matter
		mu[2] = locations[j,2] #y-coordinate #else doesn't matter

        #may need to truncate the mvnormal distribution at some point so it's within the frame,
        #but that doesn't seem to be a problem yet
		cov[1,1] = sd_x
		cov[2,2] = sd_y
		cov[1,2] = 0
		cov[2,1] = 0

		#dimensions of the frames
		low_x = 0.0 #trying to include stuff that's not in the frame
		high_x = 40.0
		low_y = 0.0
		high_y = 40.0

		#a_first = @trace(bernoulli(0.5), (:a_first, j))

        params = MBRFSParams([FA, (1-M)*E],
                             [mvuniform, mvnormal],
                             [(low_x,low_y, high_x, high_y), (mu,cov)])
        C[j] = @trace(mbrfs(params), (:C => j))
		#C[j,1] = a_first ? @trace(brfs(FA, mvuniform, (low_x,low_y, high_x, high_y)), (:C => j => 1)) : @trace(brfs((1-M)*E, mvnormal, (mu,cov)), (:C => j => 1)) #return [x, y] for location or nothing
		#C[j,2] = a_first ? @trace(brfs((1-M)*E, mvnormal, (mu,cov)), (:C => j => 2)) : @trace(brfs(FA, mvuniform, (low_x,low_y, high_x, high_y)), (:C => j => 2)) #return tuple for location or null

		#Q_A returns a tuple for location sampled from uniform distribution or null
		#Q_B returns a tuple for location sampled from gaussian distribution or null
	end #end for

	return C
end


###################################################################################################################

alpha = 2
beta = 10

#Define generative model gm. gm takes as input the possible objects, the number of percepts to produce, and the number of frames
#per percepts.
@gen function gm(possible_objects::Vector{String}, n_percepts::Int, n_frames::Int)

	#dimensions of the frames
	low_x = 0.0
	high_x = 40.0
	low_y = 0.0
	high_y = 40.0

	#need to make one possible_objects to change when replaced, another to not change?
	possible_objects_immutable = copy(possible_objects)

	#Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

	for j = 1:length(possible_objects)
        #set false alarm rate
        V[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to miss rate of around 0.1
        #set miss rate
        V[j,2] = @trace(Gen.beta(alpha, beta), (:m, j)) #leads to miss rate of around 0.1
	end

	#Determining frame of reality R
	lambda_objects = 1 #must be <= length of possible_objects
	low = -1  #seems that low is never sampled, so this distribution will go from low+1 to high
	num_categories = length(possible_objects_immutable)

	#generate each percept

    #percepts will contain many percepts.
    percepts = []

    #Rs will contain many realities
    Rs = []
	#locationses will contain many locations
	locationses = []

    for p = 1:n_percepts

        possible_objects_mutable = copy(possible_objects)

    	#numObjects = @trace(Gen.poisson(lambda_objects), (:numObjects, p))
		numObjects = @trace(trunc_poisson(lambda_objects, low, num_categories), (p => :numObjects))

		#building R
        R = @trace(sample_wo_repl(possible_objects, numObjects), (p => :R))
        push!(Rs, R)
		locations = Matrix{Float64}(undef, num_categories, 2)
		for i = 1:num_categories
			E = (possible_objects[i] in R)
			locations[i,:] = E ? @trace(mvuniform(low_x,low_y, high_x, high_y), (p => :location => i)) : @trace(mvuniform(-1.0,-1.0, -1.0, -1.0), (p => :location => i)) #gives [x,y] coordinates
		end
		push!(locationses, locations)

    	percept = []
    	for f = 1:n_frames
    		perceived_frame = @trace(build_percept(R, V, locations, possible_objects), (p => (:perceived_frame, f)))
    		push!(percept, perceived_frame)
    	end

        push!(percepts, percept)

    end

	return (Rs, locationses, V, percepts); #returning reality R, (optional)
end;
