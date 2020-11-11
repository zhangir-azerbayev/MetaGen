Base.@kwdef struct Video_Params
    lambda_objects::Float64 = 1
    possible_objects::Vector{Int64} = [1, 2, 3, 4, 5]
    v::Matrix{Float64} = zeros(5, 2)
    x_max::Float64 = 100
    y_max::Float64 = 100
end

Frame = Vector{Detection}

"""
    This function takes a category and params and it returns the possible
    objects (as Detections) of that category that could be detected
"""
@gen function gen_possible_detection(params::Video_Params, cat::Int64)
    x = @trace(uniform(0,100), :x)
    y = @trace(uniform(0,100), :y)

    println("params.v[:,2] ", params.v[:,2])
    lambda = params.v[cat,2]

    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.;0. sd_y]

    PoissonElement{Detection}(lambda, object_distribution_present,
                              ([x,y], cov, cat))
end

@gen function gen_possible_hallucination(params::Video_Params, cat::Int64)
    x = @trace(uniform(0,100), :x)
    y = @trace(uniform(0,100), :y)

    println("params.v[:,1] ", params.v[:,1])
    fa = params.v[cat,1]

    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.;0. sd_y]

    BernoulliElement{Detection}(fa, object_distribution_present,
                              ([x,y], cov, cat))
end

possible_detection_map = Gen.Map(gen_possible_detection)

possible_hallucination_map = Gen.Map(gen_possible_hallucination)

@gen function init_scene(params::Video_Params)
    println("init_scene")
    #set up the scene / T0 world state

    #saying there must be at least one object per scene, and at most 100
    numObjects = @trace(trunc_poisson(params.lambda_objects, 0.0, 100.0), (:numObjects)) #may want to truncate so 0 objects isn't possible
    c = @trace(multinomial(numObjects,[0.2,0.2,0.2,0.2,0.2]), (:c))

    paramses = fill(params, numObjects)
    @trace(possible_detection_map(paramses, c), :init_scene)
end

@gen function frame_kernel(current_frame::Int64, state, params)
    println("frame_kernel")
    #update imaginary objects

    println("params.v[:,1] ", params.v[:,1])
    sum_fas_imaginary_objects = sum(params.v[:,1])#get lambdas for absent
    numObjects = @trace(poisson(sum_fas_imaginary_objects), (:numObjects)) #may want to truncate so 0 objects isn't possible
    #numObjects = @trace(poisson(2), (:numObjects)) #may want to truncate so 0 objects isn't possible

    #normalizing
    fas_normalized = params.v[:,1]./sum_fas_imaginary_objects
    println("fas_normalized ", fas_normalized)
    c = @trace(multinomial(numObjects, fas_normalized), (:c))

    paramses = fill(params, numObjects)
    imagined_objects = @trace(possible_hallucination_map(paramses, c), :imagined_objects)

    ####Actually render the observations

    combined_obj = vcat(state, imagined_objects)
    display(params.v)
    println("length(state) ", length(state))
    println("length(imagined_objects) ", length(imagined_objects))
    #combined_obj = collect(PoissonElement{Detection}, combined_obj)
    combined_obj = RFSElements{Detection}(combined_obj)
    println("length(combined_obj) ", length(combined_obj))

    # n = length(combined_obj)
    # #many_element_rfs = collect()
    # #many_element_rfs = collect(Int64, map(i -> rand(Distributions.Categorical(probs)), 1:n))
    # many_element_rfs = RFSElements{Detection}(undef,n)
    # for i = 1:n
    #     many_element_rfs[i] = combined_obj[i]
    # end
    #observations = @trace(rfs(many_element_rfs), :observations)
    observations = @trace(rfs(combined_obj), :observations)


    return state #just keep sending the scene / initial state in.
end

frame_chain = Gen.Unfold(frame_kernel)

@gen function video_kernel(num_frames::Int64, params::Video_Params)

    println("in video kernel")

    init_state = @trace(init_scene(params), :init_state)
    states = @trace(frame_chain(num_frames, init_state, params), :frame_chain)
    return (init_state, states)
end

#video_chain = Gen.Unfold(video_kernel)
video_map = Gen.Map(video_kernel)

frame_chain = Gen.Unfold(frame_kernel)

export video_map, Video_Params
