@gen function metacog(possible_objects::Vector{Int64})

    #sample parameters
    #set up visual system's parameters
    #Determining visual system V
	v = Matrix{Float64}(undef, length(possible_objects), 2)
	alpha = 2
	beta = 10
    alpha_hit = 10
	beta_hit = 2
	for j = 1:length(possible_objects)
        #set lambda when target absent
        v[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to fa rate of around 0.1
        #set lambda when target present
        v[j,2] = @trace(Gen.beta(alpha_hit, beta_hit), (:hit, j))
	end

    params = Video_Params(v = v, possible_objects = possible_objects)

    num_frames = 3
    num_videos = 2
    fs = fill(num_frames, num_videos) #nummber of frames per video
    ps = fill(params, num_videos)

    videos = @trace(video_map(fs, ps), :videos)
end

export metacog
