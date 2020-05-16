#Inference procedure file
#include("gm_with_fragmentation.jl")
include("gm_with_FA_M_and_locations.jl")
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

#function for printing stuff to file
#needs traces, num_samples, possible_objects
function print_Vs_and_Rs_to_file(tr, num_samples, possible_objects)
    #initialize something for tracking average V
    avg_V = zeros(length(possible_objects), 2)
    #initialize something for realities
    realities = Array{Array{String}}[]
    Rs_locationses_array = Array{Matrix{Float64}}[]
    for i = 1:num_samples
        R,Rs_locationses,V,_,_ = Gen.get_retval(tr[i])
        avg_V = avg_V + V/num_samples
        push!(realities,R)
        push!(Rs_locationses_array,Rs_locationses)
        # println("R is ", R)
        # println("V is ", V)
    end
    # println("avg_V is ", avg_V)
    print(file, avg_V, " & ")
    dictionary = countmemb(realities)
    print(file, dictionary, " & ")
    dictionary_locations = countmemb(Rs_locationses_array)
    print(file, dictionary_locations, " & ")
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
function set_observations(obs, gt_choices, p, n_frames)
    for f = 1:n_frames
        #giving it visual_count and all the locations (for hallucinated and non-hallucinated objects)
        addr = p => (:perceived_frame,f) #for gm_Julian_intermediate
        sm = Gen.get_submap(gt_choices, addr)
        Gen.set_submap!(obs, addr, sm)
    end
end

#perturb each entry of V independently
#j is the index of the possible object whose hall_lambda or M will be perturbed
#hall is a boolean for if it will perturb the hall_lambda or M. If true, perturb FA. If false, perturb M.
#std controls the standard deviation of the normal perpurbations of the fa and miss rates
@gen function perturbation_proposal_individual(prev_trace, std::Float64, j::Int, hall::Bool)
    choices = get_choices(prev_trace)
    if hall
	    FA = @trace(trunc_normal(choices[(:fa, j)], std, 0.0, 1.0), (:fa, j))
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
		trace,_ = Gen.metropolis_hastings(trace, perturbation_proposal_individual, (0.1,index,isodd(i)))
	end
	return trace
end;


function particle_filter(num_particles::Int, gt_percepts, gt_percepts_locationses, gt_choices, num_samples::Int)

	n_percepts = size(gt_percepts)[1]
	n_frames = size(gt_percepts[1])[1]

	println("in particle_filter")

	# construct initial observations
	init_obs = Gen.choicemap()
	p=1
    set_observations(init_obs, gt_choices, p, n_frames)

	println("init_obs")
	display(init_obs)

	#initial state
	#num_percepts is 1 because starting off with just one percept
	state = Gen.initialize_particle_filter(gm, (possible_objects, 1, n_frames), init_obs, num_particles)

	for p = 2:n_percepts

		println("percept ", p-1)
        print_weights(state, "at start of loop is ")
        print_ess(state, "ess at start of loop is ")


		# #tr = Gen.sample_unweighted_traces(state, num_samples)
		# tr = get_traces(state)
        #
		# return a sample of unweighted traces from the weighted collection
		tr = Gen.sample_unweighted_traces(state, num_samples)

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

        print_weights(state, "after perturbation ")
        print_ess(state, "after perturbation ")

		do_resample = Gen.maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)

        print_weights(state, "after resample ")
        print_ess(state, "after resample ")

		obs = Gen.choicemap()
		set_observations(obs, gt_choices, p, n_frames)

        # println("obs")
    	# display(obs)

        print_weights(state, "after setting observation ")
        print_ess(state, "after setting observation ")

		Gen.particle_filter_step!(state, (possible_objects, p, n_frames), (UnknownChange(),), obs)

        print_weights(state, "after particle filter step ")
        print_ess(state, "after particle filter step ")

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

    print_Vs_and_Rs_to_file(tr, num_samples, possible_objects)

	return tr
end;

##########################
