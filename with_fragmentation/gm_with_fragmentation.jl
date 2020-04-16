#Generative model with fragmentation file

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


#small issue where all of the inputs need to by Float64. Doesn't accept Int64s
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
		else
			detection_count = 0
		end #end if
		#if object isn't in reality, visual_count only depends on hallucination_count
		sum_count = max(detection_count + hallucination_count, 1.0E-4)
		visual_count = @trace(trunc_poisson(sum_count, -1.0, hard_cap), (:visual_count => j))

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

        R = @trace(sample_with_repl(possible_objects_mutable, numObjects), (:R, p))
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
