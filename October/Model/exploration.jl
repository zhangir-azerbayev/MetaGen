using Gen
using GenRFS

a =  RFSElements{Float64}(undef, 1)

lambda_objects = 1
probs = [0.2, 0.2, 0.2, 0.2, 0.2]
p = PoissonElement{Float64}(lambda_objects, categorical, (probs))

a[1] = p
rfs(a)

############

a =  RFSElements{Float64}(undef, 2)
lambda_objects = 4
p1 = PoissonElement{Float64}(lambda_objects, normal, (10., 1.))
p2 = PoissonElement{Float64}(lambda_objects, normal, (-10., 1.))
a[1] = p1
a[2] = p2
rfs(a)

############

@gen function gm(val)
    a =  RFSElements{Float64}(undef, 2)
    lambda_objects = 4
    p1 = PoissonElement{Float64}(lambda_objects, normal, (10., 1.))
    p2 = PoissonElement{Float64}(lambda_objects, normal, (-10., 1.))
    a[1] = p1
    a[2] = p2

    objs = @trace(rfs(a), (:obj))

    return objs
end

gt_trace,_ = Gen.generate(gm, (1,))
objs = Gen.get_retval(gt_trace)
gt_choices = get_choices(gt_trace)
