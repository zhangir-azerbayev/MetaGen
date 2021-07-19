@gen (static) function lambda_fa(arg1::Real, arg2::Real)
	fa = @trace(gamma(arg1, arg2), :fa)
	return fa
end

@gen (static) function miss_rate(arg1::Real, arg2::Real)
	miss = @trace(beta(arg1, arg2), :miss)
	return miss
end

"""
	make_visual_system(params::Video_Params)

Samples the matrix describing the initial visual system.
"""
@gen (static) function make_visual_system(params::Video_Params)
	#alpha = 2
	#beta = 10
	#alpha_hit = 10
	#beta_hit = 2
	#for j = 1:length(params.possible_objects)
		#set lambda when target absent
		#v[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to fa rate of around 0.1
		#v[j,1] = @trace(trunc_normal(0.002, 0.005, 0.0, 1.0), (:lambda_fa, j)) #these are lambdas per receptive field
	arg1 = fill(1.0, length(params.possible_objects))
	arg2 = fill(1.0, length(params.possible_objects))
	fa = @trace(Map(lambda_fa)(arg1, arg2), :lambda_fa) #these are lambdas per receptive field
		#v[j,1] = @trace(uniform(0.0, 1.0), (:lambda_fa, j))
		#set miss rate when target present
		#v[j,2] = @trace(trunc_normal(0.25, 0.5, 0.0, 1.0), (:miss_rate, j))

	miss = @trace(Map(miss_rate)(arg1, arg2), :miss_rate)

	v = hcat(fa, miss)
   	return convert(Matrix{Real}, v)
end
################################################################################

"""
	main(num_videos::Int64, num_frames::Int64)

Simulates the visual system, `num_videos` scenes each with
`num_frames` frames, and percepts generated by the visual system.
"""
@gen (static) function main(num_videos::Int64, num_frames::Int64)

	params = Video_Params()

    #sample parameters
    #set up visual system's parameters
    #Determining visual system V


	# n = length(params.possible_objects)
	# v = Matrix{Real}(undef, n, 2)

	#v[:,1] = @trace(Map(trunc_normal(0.002, 0.005, 0.0, 1.0), collect(1:n)), :lambda_fa)
	#v[:,2] = @trace(Map(trunc_normal(0.25, 0.5, 0.0, 1.0), collect(1:n)), :miss_rate)

	#temp = @trace(Map(foo)(collect(1:n)), :lambda_fa)

	receptive_fields = make_receptive_fields()

	params = Video_Params(num_receptive_fields = length(receptive_fields))

    #fs = fill(num_frames, num_videos) #number of frames per video
    #ps = fill(params, num_videos)
	#vs = fill(v, num_videos)
	#receptive_fieldses = fill(receptive_fields, num_videos)

	init_v_matrix = @trace(make_visual_system(params), :init_v_matrix)
	alphas = fill(1, size(init_v_matrix))
	betas = fill(1, size(init_v_matrix))
	v_matrix_state = (init_v_matrix, alphas, betas)
	#println("alphas in main ", v_matrix_state[2])
    videos = @trace(video_chain(num_videos, v_matrix_state, num_frames, params, receptive_fields), :videos)
	return videos
end

export main
export make_visual_system
