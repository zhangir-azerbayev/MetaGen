using Gen

@gen function foo()
    v = Matrix{Real}(undef, 5, 2)
    for j = 1:5
        v[j,1] = @trace(uniform(0.0, 1.0), (:lambda_fa, j))
        v[j,2] = @trace(uniform(0.0, 1.0), (:miss_rate, j))
    end
    return v
end

@gen (static) function foo2()
    v = Matrix{Real}(undef, 5, 2)
    v = @trace(foo(), :v_matrix)

    obs1 = @trace(uniform(0.0, v[1, 1]), :obs1)
    obs2 = @trace(uniform(0.0, v[1, 2]), :obs2)
end


function inference(trace)
    j = 1
    for i=1:100
        trace, accepted = hmc(trace, select(:v_matrix => (:miss_rate, j)))
    end
end

@load_generated_functions
#foo2()

gt_trace,_ = Gen.generate(foo2, ())
#println(gt_trace)
# gt_choices = get_choices(gt_trace)
#
# obs = Gen.choicemap()
# obs[:obs1] = gt_choices[:obs1]
# obs[:obs2] = gt_choices[:obs2]
#
inference(gt_trace)
