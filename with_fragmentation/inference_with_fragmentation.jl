#Inference procedure file
include("gm_with_fragmentation.jl")

using Gen
using FreqTables
using Distributions
using Distances
using TimerOutputs


###################################################################################################################
#function for shuffling elements of an array
function homebrew_shuffle(a::AbstractArray)
    n = length(a)
    new_array = []
    memory_array = trues(n) #index hasn't been picked yet

    while length(new_array) < (n-1)
        candidate = rand(1:n)
        if memory_array[candidate]
            push!(new_array, candidate)
            memory_array[candidate] = false
        end
    end
    #add the last one
    last = findfirst(memory_array)
    push!(new_array, last)
    return new_array
end



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
		upper_bound = 100.0 #make sure it matches upper bound in gm
    	hall = @trace(trunc_normal(choices[(:hall_lambda, j)], std, 0.0, upper_bound), (:hall_lambda, j))
    else
    	#new M rate will be between 0 and 1
        M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 1.0), (:m, j))
    end
end

# If I allowed a resample of V, that would defeat the purpose of posterior becoming new prior.
# Instead, just add some noise.
function perturbation_move(trace)

	#Choose order of perturbation proposals randomly
	#mix up the order of the permutations
	#2 * for FA and M
    mixed_up = collect(1:2*length(possible_objects))
    mixed_up = homebrew_shuffle(mixed_up)
	for j = 1:length(mixed_up)
		i = mixed_up[j]
		index = floor((i+1)/2)
		trace, _ = Gen.metropolis_hastings(trace, perturbation_proposal_individual, (0.1,index,isodd(i)))
	end
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

	println("init_obs")
	display(init_obs)

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

		# if isnan(ess)
		# 	t = filter(t -> isinf(get_score(t)), state.traces)
		# 	ts = map(Gen.get_choices, t)
        #     println("ts[1]")
		# 	display(ts[1])
        #     println("ts[2]")
		# 	display(ts[2])
		# end




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
			# println("R is ", R)
			# println("V is ", V)
		end
		# println("avg_V is ", avg_V)
		print(file, avg_V, " & ")


		# apply rejuvenation/perturbation move to each particle. optional.
        for i = 1:num_particles
        	R,V,_ = Gen.get_retval(state.traces[i])
        	# println("V before perturbation ", V)

            state.traces[i] = perturbation_move(state.traces[i])

            R,V,_ = Gen.get_retval(state.traces[i])
            #println("R after perturbation is ", R)
			# println("V after perturbation ", V)
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

		# if isnan(ess)
		# 	t = filter(t -> isinf(get_score(t)), state.traces)
		# 	ts = map(Gen.get_choices, t)
		# 	println("ts[1]")
		# 	display(ts[1])
        #     println("ts[2]")
		# 	display(ts[2])
		# end
	end
	# return a sample of unweighted traces from the weighted collection
	tr = Gen.sample_unweighted_traces(state, num_samples)

	println("percept ", n_percepts)

	#initialize something for tracking average V
	avg_V = zeros(length(possible_objects), 2)
	for i = 1:num_samples
		R,V,_ = Gen.get_retval(tr[i])
		avg_V = avg_V + V/num_samples
		# println("R is ", R)
		# println("V is ", V)
	end
	print(file, avg_V, " & ")

	return tr
end;

##########################
