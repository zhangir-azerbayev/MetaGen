# #proposal
# @gen function foo(trace)
#     scene = trace[:init_scene]
#     n = length(scene)
#
#     id = {:id} ~ categorical(fill(1/n, n))
#     others = setdiff(1:n, id)
#
#     changed = helper(es[id], 0.5, 10.) #maybe remove
#
#     #maybe make poisson
#     to_add = BernoulliElement{Object3D}(0.5, object_distribution, (params,))#make p for bernoulli and argment to foo
#
#     vec = cat(changed, to_add, dims=1)
#     println("length(vec) ", length(vec)) #I feel like something is wrong with using params
#     vec = RFSElements{Object3D}(vec)
#
#     {:set} ~ rfs(vec)
# end
#
# @transform foo_involution (model_in, aux_in) to (model_out, aux_out) begin
#
#     scene = @read(model_in[:init_scene], :discrete)
#     id = @read(aux_in[:id], :discrete)
#     n = length(scene)
#     others = scene[setdiff(1:n, id)]
#     set = @read(aux_in[:set], :discrete)
#     k = length(set)
#
#     if k==0
#
#     end
#
#     vec = cat(others, set, dims=1)
#
#     @write(model_out[:init_scene], vec, :discrete)
#
#
#     @write(aux_out[:set], scene[id:id], discrete) #
#
# end

#perturb_params
Base.@kwdef struct Perturb_Params
    probs_possible_objects::Vector{Float64}
end



#proposal. like split_merge_proposal.
@gen function add_remove_or_change_proposal(trace, v::Int64, variance::Float64, perturb_params::Perturb_Params)
    scene = trace[:videos => v => :init_scene]
    n = length(scene)

    #if the scene is already empty, can only add.
    if n<1
        edit_type = {:edit_type} ~ categorical([1.0])
    else
        edit_type = {:edit_type} ~ categorical(fill(1/3, 3))
    end

    #if add
    if edit_type==1
        params2 = Video_Params(probs_possible_objects = perturb_params.probs_possible_objects)
        new = {:new} ~ object_distribution(params2)

    #remove
    elseif edit_type==2
        id = {:id} ~ categorical(fill(1/n, n)) #select element to remove

    #change location
    elseif edit_type==3
        id = {:id} ~ categorical(fill(1/n, n)) #select element to change
        new = {:new} ~ object_distribution_present([scene[id][1], scene[id][2], scene[id][3]], diagm([variance, variance, variance]), scene[id][4])
    end
end

@transform involution (t, u) to (t_prime, u_prime) begin

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
    else
        id = @read(u[:id], :discrete)
        e = @read(u[:new], :discrete)
        new_scene = deepcopy(scene)
        e_old = scene[id]
        new_scene[id] = e
        @write(t_prime[:videos => v => :init_scene], new_scene, :discrete)
        @write(u_prime[:edit_type], 3, :discrete)
        @write(u_prime[:id], id, :discrete)
        @write(u_prime[:new], e_old, :discrete)
    end

    #@write(u_prime[:v], v, :discrete)
end

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

add_remove_or_change_kernel(trace, v, variance, perturb_params) = mh_here(trace, add_remove_or_change_proposal, (v, variance, perturb_params), involution)
