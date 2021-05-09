
#proposal. like split_merge_proposal.
@gen function add_remove_proposal(trace, v::Int64, line_segments::Array{Line_Segment}, perturb_params::Perturb_Params)
    scene = trace[:videos => v => :init_scene]
    n = length(scene)

    #if the scene is already empty, can only add.
    if n<1
        edit_type = {:edit_type} ~ categorical([1.0])
    else
        edit_type = {:edit_type} ~ categorical(fill(1/2, 2))
    end

    #if add
    if edit_type==1
        params2 = Video_Params(probs_possible_objects = perturb_params.probs_possible_objects)
        new = {:new} ~ new_object_distribution(params2, line_segments)

        #remove
    elseif edit_type==2
        id = {:id} ~ categorical(fill(1/n, n)) #select element to remove
    end
end

@transform add_remove_involution (t, u) to (t_prime, u_prime) begin

    #v = @read(u[:v], :discrete)
    _,v,_,_ = get_args(u)

    scene = @read(t[:videos => v => :init_scene], :discrete)
    n = length(scene)

    edit_type = @read(u[:edit_type], :discrete)

    #if add
    if edit_type==1
        e = @read(u[:new], :discrete)
        new_scene = deepcopy(scene)
        temp = push!(new_scene, e)
        #println("temp ", temp)
        #push!(scene, e)
        @write(t_prime[:videos => v => :init_scene], new_scene, :discrete)#is the problem here?
        @write(u_prime[:edit_type], 2, :discrete)
        @write(u_prime[:id], n+1, :discrete)
        #if remove
    elseif edit_type==2
        id = @read(u[:id], :discrete)
        new_scene = deepcopy(scene)
        e = scene[id]
        deleteat!(new_scene, id)
        #println("temp ", temp)

        @write(t_prime[:videos => v => :init_scene], new_scene, :discrete)
        @write(u_prime[:edit_type], 1, :discrete)
        @write(u_prime[:new], e, :discrete)

        #@write(u_prime[:v], v, :discrete)
    end
end

@gen function change_location_proposal(trace, v::Int64, variance::Float64, perturb_params::Perturb_Params)
    scene = trace[:videos => v => :init_scene]
    n = length(scene)

    id = {:id} ~ categorical(fill(1/n, n)) #select element to change
    new = {:new} ~ object_distribution_present([scene[id][1], scene[id][2], scene[id][3]], diagm([variance, variance, variance]), scene[id][4])
end

@gen function change_category_proposal(trace, v::Int64, perturb_params::Perturb_Params)
    scene = trace[:videos => v => :init_scene]
    n = length(scene)

    id = {:id} ~ categorical(fill(1/n, n)) #select element to change
    new = {:new} ~ object_distribution_category(scene[id], perturb_params)
    #TODO write object_distribution_category
end

@transform change_involution (t, u) to (t_prime, u_prime) begin

#v = @read(u[:v], :discrete)
_,v,_,_ = get_args(u)

scene = @read(t[:videos => v => :init_scene], :discrete)
n = length(scene)

id = @read(u[:id], :discrete)
e = @read(u[:new], :discrete)
new_scene = deepcopy(scene)
e_old = scene[id]
new_scene[id] = e
@write(t_prime[:videos => v => :init_scene], new_scene, :discrete)
@write(u_prime[:id], id, :discrete)
@write(u_prime[:new], e_old, :discrete)
end

add_remove_kernel(trace, v, variance, perturb_params) = mh_here(trace, add_remove_proposal, (v, line_segments, perturb_params), add_remove_involution)
change_location_kernel(trace, v, variance, perturb_params) = mh_here(trace, change_location_proposal, (v, variance, perturb_params), change_involution)
change_category_kernel(trace, v, variance, perturb_params) = mh_here(trace, change_category_proposal, (v, perturb_params), change_involution)


function metropolis_hastings(
    trace, proposal::GenerativeFunction,
    proposal_args::Tuple, involution::Union{TraceTransformDSLProgram,Function};
    check=false, observations=EmptyChoiceMap())
    trace_translator = SymmetricTraceTranslator(proposal, proposal_args, involution)
    (new_trace, log_weight) = trace_translator(trace; check=check, observations=observations)
    println("proposal ", new_trace[:videos => 1 => :init_scene])
    if log(rand()) < log_weight
        # accept
        (new_trace, true)
    else
        # reject
        (trace, false)
    end
end

function metropolis_hastings_here(
    trace, proposal::GenerativeFunction,
    proposal_args::Tuple, involution::Union{TraceTransformDSLProgram,Function};
    check=false, observations=EmptyChoiceMap())
    trace_translator = SymmetricTraceTranslator(proposal, proposal_args, involution)
    (new_trace, log_weight) = trace_translator(trace; check=check, observations=observations)
    println("proposal ", new_trace[:videos => 1 => :init_scene])
    if log(rand()) < log_weight
        # accept
        (new_trace, true)
    else
        # reject
        (trace, false)
    end
end

const mh_here = metropolis_hastings_here
