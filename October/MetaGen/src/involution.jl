
"""
    add_remove_proposal(trace, v::Int64, line_segments_per_category::Array{Array{Line_Segment,1},1}, perturb_params::Perturb_Params)

Proposes adding or removing an object.

Proposal function used for HMC. like split_merge_proposal.
"""
@gen function add_remove_proposal(trace, v::Int64, line_segments_per_category::Array{Array{Line_Segment,1},1}, perturb_params::Perturb_Params)
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
        new = {:new} ~ new_object_distribution_noisy_or_uniform(params2, line_segments_per_category)
        #println("new ", new)
        #remove
    elseif edit_type==2
        id = {:id} ~ categorical(fill(1/n, n)) #select element to remove
        #println("remove ", scene[id])
    end
end
"""
    add_remove_involution (t, u) to (t_prime, u_prime)

Involution corresponding to `add_remove_proposal`
"""
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

"""
    change_category_proposal(trace, v::Int64, perturb_params::Perturb_Params)

Proposes changing an object category according to the distribution provided
in `perturb_params`.
"""
@gen function change_category_proposal(trace, v::Int64, perturb_params::Perturb_Params)
    scene = trace[:videos => v => :init_scene]
    n = length(scene)

    id = {:id} ~ categorical(fill(1/n, n)) #select element to change
    #make it so doesn't propose current category
    probs_possible_objects = deepcopy(perturb_params.probs_possible_objects)
    probs_possible_objects[id] = 0
    probs_possible_objects = probs_possible_objects./sum(probs_possible_objects)
    perturb_params_new = Perturb_Params(probs_possible_objects)

    new = {:new} ~ object_distribution_category(scene[id][1], scene[id][2], scene[id][3], perturb_params_new)
end

```
    change_location_involution (t, u) to (t_prime, u_prime) begin

Involution corresponding to `change_location_proposal`
```
@transform change_location_involution (t, u) to (t_prime, u_prime) begin

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

"""
    change_category_involution (t, u) to (t_prime, u_prime)

Involution corresponding to `change_category_proposal`
"""
@transform change_category_involution (t, u) to (t_prime, u_prime) begin

    #v = @read(u[:v], :discrete)
    _,v,_, = get_args(u)

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

add_remove_kernel(trace, v, line_segments, perturb_params) = mh(trace, add_remove_proposal, (v, line_segments, perturb_params), add_remove_involution)
change_location_kernel(trace, v, variance, perturb_params) = mh(trace, change_location_proposal, (v, variance, perturb_params), change_location_involution)
change_category_kernel(trace, v, perturb_params) = mh(trace, change_category_proposal, (v, perturb_params), change_category_involution)

"""
Duplicated from the Gen library
"""
function metropolis_hastings_here(
    trace, proposal::GenerativeFunction,
    proposal_args::Tuple, involution::Union{TraceTransformDSLProgram,Function};
    check=false, observations=EmptyChoiceMap())
    trace_translator = SymmetricTraceTranslator(proposal, proposal_args, involution)
    (new_trace, log_weight) = trace_translator(trace; check=check, observations=observations)
    println("proposal ", new_trace[:videos => 1 => :init_scene])
    println("ratio ", exp(log_weight))
    if log(rand()) < log_weight
        # accept
        (new_trace, true)
    else
        # reject
        (trace, false)
    end
end

const mh_here = metropolis_hastings_here
