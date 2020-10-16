using Gen
using GenRFS
include("custom_distributions.jl")

@gen function generatitive_model(possible_objects::Vector{String})

    #set up visual system's parameters
    #Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

    shape_a = 0.5
    scale_a = 1.0 #very close to 0
    shape_p = 2
    scale_p = 1 #mean is 2, mode is 1
	for j = 1:length(possible_objects)
        #set lambda when targent absent
        V[j,1] = @trace(Gen.gamma(shape_a, scale_a), (:lam_absent, j)) #leads to miss rate of around 0.1
        #set lambda when targent present
        V[j,2] = @trace(Gen.gamma(shape_p, scale_p), (:lam_present, j)) #leads to miss rate of around 0.1
	end




    #set up T0 world state

    lambda_objects = 1
    #numObjects = @trace(poisson(lambda_objects), (:numObjects)) #may want to truncate so 0 objects isn't possible
    numObjects = 1

    #c = @trace(Multinomial(numObjects, [0.2,0.2,0.2,0.2,0.2]), (:c))
    c = [2]

    for i = 1:length(c)
        x = @trace(uniform(0,100), (i ,:x))
        y = @trace(uniform(0,100), (i ,:y))

        lambda_a = V[c[i],2]

        sd_x = 10
        sd_y = 10
        object = PoissonElement{Float64}(lambda_a, objectdistribution, ([x,y], [sd_x 0;0 sd_y], c[i]))

    end
end










possible_objects = ["person","bicycle","car","motorcycle","airplane"]
#call it
gt_trace,_ = Gen.generate(generative_model, (possible_objects,))
gt_choices = get_choices(gt_trace)
