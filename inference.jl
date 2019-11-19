#using PyCall

using Gen
using Distributions

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
    sample = Array{String}(undef,n)
    for i in 1:n
    	idx = @trace(Gen.uniform_discrete(1, length(A)), (:idx, i))
        sample[i] = splice!(A, idx)
    end
    return sample
end

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
		IDs[i] = findfirst(isequal(names[i]),class_names)
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
		V[j,1] = @trace(Gen.beta(1.909091, 107.99999999999999), (:fa, j)) #leads to false alarm rate of 0.01
		#set miss rate
		V[j,2] = @trace(Gen.beta(1.909091, 36.272727), (:m, j)) #leads to miss rate of 0.05
	end

	#Determining frame of reality R
	lambda = 5 #must be <= length of possible_objects
	low = 1
	high = length(possible_objects_immutable)

	numObjects = @trace(trunc_poisson(lambda, low, high), :numObjects)
    R = @trace(sample_wo_repl(possible_objects_mutable,numObjects), :R)

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


	return (R,percept) #returning reality R, (optional)
end;


##############################################################################################
#Defining observations / constraints

possible_objects = ["person", "bicycle", "car","motorcycle", "airplane"]
#Later, possible_objects will equal class_names

# define some observations
gt = Gen.choicemap()

# #VTrue is the real Visual system
# VTrue = Matrix{float}(undef, length(possible_objects), 2)
# #for loop to make V
# for j = 1:length(possible_objects)
# 		#set false alarm rate
# 		VTrue[j,1] = @trace(Gen.beta(1.909091, 107.99999999999999), :fa, j) #leads to false alarm rate of 0.01
# 		gt[:fa, j] = VTrue[j,1]
# 		#set miss rate
# 		VTrue[j,2] = @trace(Gen.beta(1.909091, 36.272727), :m, j) #leads to miss rate of 0.05
# 		gt[:m, j] = VTrue[j,2]
# 	end


#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
n_frames = 10
gt_trace,_ = Gen.generate(gm, (possible_objects, n_frames))
gt_reality,gt_percept = Gen.get_retval(gt_trace)

println("gt_reality is ",gt_reality)
println("gt_percept is ",gt_percept) #could translate back into names

# #Translating gt_percept back into names
# percept = Matrix{String}(undef, n_frames, length(possible_objects))
# for f = 1:n_frames
# 	percept[f,:] = possible_objects[gt_percept[f,:]]
# end
# println("percept is ",percept)

gt_choices = Gen.get_choices(gt_trace)
println(gt_choices)


#get the percepts
obs = Gen.get_submap(gt_choices, :percept)
#initialize a new trace
trace, _ = Gen.generate(gm, (possible_objects, n_frames), obs)

# inference_history = Vector{typeof(trace)}(undef, N)
# for i = 1:N
# 	#selection = Gen.select(:fa, :m, :numObjects, :obj_selection)
# 	trace,_ = Gen.hmc(trace)
# 	inference_history[i] = trace
# end