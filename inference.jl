#using PyCall

using Gen

#push!(pyimport("sys")["path"], pwd());
#pyimport("something.py")[:hello_world]()
#pythonFile = pyimport("something")

#pythonFile.hello_world()

struct TruncatedPoisson <: Gen.Distributions{Int} end

const trunc_poisson = TruncatedPoisson()

function logpdf(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U) where {U <: Real}
	d = Distributions.Poisson(lambda)
	td = Distributions.Truncated(d, low, high)
    Distributions.logpdf(td, x)
end

function logpdf_grad(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U)  where {U <: Real}
    gerror("Not implemented")
    (nothing, nothing)
end

function random(::TruncatedPoisson, lambda::U, low::U, high::U) where {U <: Real}
	d = Distributions.Poisson(lambda)
    rand(Distributions.Truncated(d, low, high)
end

(::TruncatedPoisson)(lambda, low, high) = random(TruncatedPoisson(), lambda, low, high)
is_discrete(::TruncatedPoisson) = true

has_output_grad(::TruncatedPoisson) = false
has_argument_grads(::TruncatedPoisson) = (false,)


struct Frame
	objects::Vector{String}
end
# Frame(your-list)

@gen function sample_wo_repl(A,n)
    sample = Array{eltype(A)}(n)
    for i in 1:n
    	idx = @trace(Gen.uniform_discrete(1, length(A)), (:idx, i))
        sample[i] = splice!(A, idx)
    end
    return sample
end

#define generative model gm
@gen function gm(possible_objects::Vector{String}, n_frames::Int)
	fa = @trace(Gen.beta(1, 2), :fa)
	m = @trace(Gen.beta(11, 22), :m)

	lambda = 5
	low = 1
	high = 81

	numObjects = @trace(TruncatedPoisson(lambda, low, high), :numObjects)
    F = @trace(sample_wo_repl(class_names,numObjects), :obj_selection)



	percept = Matrix{Bool}(undef, t, length(possible_objects))
	for f = 1:n_frames
		for j = 1:length(possible_objects)
			percept[f,j] = @trace(bernoulli(fa), (:percept, f, j))
		end
	end
	return F
        #if @trace(bernoulli(V[names_to_IDs_single_frame(frame)[i],1]), :data => i => :is_detected)
        #    P.append(frame[i])
end

# define some observations
gt = Gen.choicemap()
gt[:fa] = 1
gt[:m] = 11
gt_trace, _ = Gen.generate(gm, (possible_objects, 10), gt)
gt_reality = Gen.get_retval(gt_trace)
gt_choices = Gen.get_choices(gt_trace)
obs = Gen.get_submap(gt_choices, :percept)


trace, _ = Gen.generate(gm, (xs,), obs)
inference_history = Vector{typeof(trace)}(undef, N)

for i = 1:N
	#selection = Gen.select(:fa, :m, :numObjects, :obj_selection)
	trace,_ = Gen.hmc(trace)
	inference_history[i] = trace
end