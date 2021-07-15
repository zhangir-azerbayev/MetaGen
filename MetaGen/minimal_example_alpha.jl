#tried to produce a minimal example of the issue with regeneration
#and alphas in video.jl, but it works here.

using Gen

@gen (static) function inner_kernel(iter::Int64, state)
    r = state[1]
    alpha = state[2]
    s = @trace(uniform(r, alpha), :s)
    alpha = alpha + s #update alpha
    new_state = (r, alpha)
    return new_state
end

inner_chain = Gen.Unfold(inner_kernel)

@gen function outer_kernel(iter::Int64, outer_state)
    alpha = outer_state[1]
    println("alpha in ", alpha)
    r = @trace(uniform(0.0, alpha), :r)
    init_inner_state = (r, alpha)
    new_inner_state = @trace(inner_chain(3, init_inner_state), :inner_chain)
    println("new_inner_state ", new_inner_state)
    alpha = new_inner_state[end][2]
    new_outer_state = (alpha,)
    println("alpha out ", alpha)
    return new_outer_state
end

outer_chain = Gen.Unfold(outer_kernel)

#@gen (static) function main()
@gen (static) function main()
    alpha = 10.0
    state = (alpha,)
    total_num_iter = 5
    @trace(outer_chain(total_num_iter, state), :outer_chain)
end

@load_generated_functions

cm = choicemap()
cm[:outer_chain => 5 => :inner_chain => 3 => :s] = 1000
gt_trace,_ = Gen.generate(main, (), cm);
get_choices(gt_trace)

new_trace,_ = regenerate(gt_trace, select(:outer_chain => 5 => :inner_chain => 3 => :s));
#works when main is not static
#when main is static, doesn't print anything
get_choices(new_trace)
