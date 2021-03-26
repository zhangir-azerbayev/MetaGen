@gen function metacog(possible_objects::Vector{Int64}, num_videos::Int64, num_frames::Int64)

    #sample parameters
    #set up visual system's parameters
    #Determining visual system V
	v = Matrix{Float64}(undef, length(possible_objects), 2)
	#alpha = 2
	#beta = 10
    #alpha_hit = 10
	#beta_hit = 2
	for j = 1:length(possible_objects)
        #set lambda when target absent
        #v[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to fa rate of around 0.1
		v[j,1] = @trace(trunc_normal(0.2, 0.5, 0.0, 10.0), (:lambda_fa, j))
        #set lambda when target present
        v[j,2] = @trace(trunc_normal(1.2, 3.0, 0.0, 10.0), (:lambda_hit, j))
	end

    permanent_camera_params = Permanent_Camera_Params()

	#square receptive fields
	pixels = 80
	n_horizontal = convert(Int64, permanent_camera_params.image_dim_x / pixels)
	n_vertical = convert(Int64, permanent_camera_params.image_dim_y / pixels)
	n = n_horizontal*n_vertical

	receptive_fields = Vector{Receptive_Field}(undef, n) #of length n
	for h = 1:n_horizontal
		for v = 1:n_vertical
			receptive_fields[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1)*pixels, (v-1)*pixels), p2 = (h*pixels, v*pixels))
		end
	end

	params = Video_Params(v = v, possible_objects = possible_objects, num_receptive_fields = n)

    fs = fill(num_frames, num_videos) #number of frames per video
    ps = fill(params, num_videos)
	qs = fill(permanent_camera_params, num_videos)
	receptive_fieldses = fill(receptive_fields, num_videos)

    videos = @trace(video_map(fs, ps, qs, receptive_fieldses), :videos)
end

export metacog
