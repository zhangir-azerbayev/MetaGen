#Inference procedure file
#include("gm_with_fragmentation.jl")
include("gm_with_FA_M_with_variable_frames.jl")
#include("gm_Julian_intermediate.jl")
include("shared_functions.jl") #just want countmemb function

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

#for printing the ess of a state with some words
#returns ess
function print_ess(state, words)
    (log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    ess = effective_sample_size(log_normalized_weights)
    println(words, ess)
    return ess
end

#for printing the log weights and log_ml_estimate of a state with some words
function print_weights(state, words)
    log_weights = get_log_weights(state)
    println("log_weights ", words, log_weights)
    log_ml_estimate = Gen.log_ml_estimate(state)
    println("log_ml_estimate ", words, log_ml_estimate)
end

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

#obs is a choicemap to add the observations to
#gt_choices are the ground truth choices as a choicemap
function set_initial_observations(obs, gt_choices, p, n_frames, V::Matrix{Float64})
    for f = 1:n_frames
        addr = (:perceived_frame,p,f) => :visual_count #for old model versionte
        sm = Gen.get_submap(gt_choices, addr)
        Gen.set_submap!(obs, addr, sm)

        for j = 1:length(possible_objects)
            obs[(:fa, j)] = V[j,1]
            obs[(:m, j)] = V[j,2]
        end
    end
end

#obs is a choicemap to add the observations to
#gt_choices are the ground truth choices as a choicemap
function set_observations(obs, gt_choices, p, n_frames)
    for f = 1:n_frames
        addr = (:perceived_frame,p,f) => :visual_count #for old model version
        sm = Gen.get_submap(gt_choices, addr)
        Gen.set_submap!(obs, addr, sm)
    end
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
		#upper_bound = 100.0 #make sure it matches upper bound in gm
    	#hall = @trace(trunc_normal(choices[(:hall_lambda, j)], std, 0.0, upper_bound), (:hall_lambda, j))
        #FA = @trace(trunc_normal(choices[(:fa, j)], std, 0.0, 1.0), (:fa, j))
        FA = @trace(trunc_normal(choices[(:fa, j)], std, 0.0, 0.5), (:fa, j))

    else
        #new M rate will be between 0 and 1
        #M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 1.0), (:m, j))
        M = @trace(trunc_normal(choices[(:m, j)], std, 0.0, 0.5), (:m, j))
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
		trace,_ = Gen.metropolis_hastings(trace, perturbation_proposal_individual, (0.1,index,isodd(i)))
	end
	return trace
end;


#if lesion is true, than set V in initial observation
function particle_filter(num_particles::Int, n_percepts, n_frames, gt_choices, num_samples::Int, Vs::Matrix{Float64}, lesion::Bool)

	println("in particle_filter")

	# construct initial observations
	init_obs = Gen.choicemap()
	p=1
    n_frames_in_this_percept = n_frames[p]
    lesion ? set_initial_observations(init_obs, gt_choices, p, n_frames_in_this_percept, V) : set_observations(init_obs, gt_choices, p, n_frames_in_this_percept)

	println("init_obs")
	display(init_obs)

	#initial state
	#num_percepts is 1 because starting off with just one percept
	state = Gen.initialize_particle_filter(gm, (possible_objects, 1, n_frames_in_this_percept), init_obs, num_particles)

	for p = 2:n_percepts

		println("percept ", p-1)

        # print_weights(state, "at start of loop is ")
        # print_ess(state, "ess at start of loop is ")

		# return a sample of unweighted traces from the weighted collection
		tr = Gen.sample_unweighted_traces(state, num_samples)

        if lesion==false
            print_Vs_and_Rs_to_file(tr, num_samples, possible_objects)
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

            # print_weights(state, "after perturbation ")
            # print_ess(state, "after perturbation ")
        end #end if lesion==false

		do_resample = Gen.maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)

        # print_weights(state, "after resample ")
        # print_ess(state, "after resample ")
        n_frames_in_this_percept = n_frames[p]
		obs = Gen.choicemap()
		set_observations(obs, gt_choices, p, n_frames_in_this_percept)

        # println("obs")
    	# display(obs)

        # print_weights(state, "after setting observation ")
        # print_ess(state, "after setting observation ")

		Gen.particle_filter_step!(state, (possible_objects, p, n_frames[1:p]), (UnknownChange(),), obs)

        # print_weights(state, "after particle filter step ")
        # print_ess(state, "after particle filter step ")

		# if isnan(ess)
		# 	t = filter(t -> isinf(get_score(t)), state.traces)
		# 	ts = map(Gen.get_choices, t)
		# 	println("ts[1]")
		# 	display(ts[1])
        #     println("ts[2]")
		# 	display(ts[2])
		# end
	end

    println("percept ", n_percepts)
	# return a sample of unweighted traces from the weighted collection
	tr = Gen.sample_unweighted_traces(state, num_samples)

    if lesion==false
        print_Vs_and_Rs_to_file(tr, num_samples, possible_objects)
    end

	return tr
end;

##########################
