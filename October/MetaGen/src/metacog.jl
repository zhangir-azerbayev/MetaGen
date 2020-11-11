@gen function metacog(possible_objects::Vector{Int64})

    #sample parameters
    #set up visual system's parameters
    #Determining visual system V
	v = Matrix{Float64}(undef, length(possible_objects), 2)
	alpha = 2
	beta = 10
    shape_p = 2
    scale_p = 1 #mean is 2, mode is 1
	for j = 1:length(possible_objects)
        #set lambda when target absent
        v[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to fa rate of around 0.1
        #set lambda when target present
        v[j,2] = @trace(Gen.gamma(shape_p, scale_p), (:lam_present, j))
	end

    params = Video_Params(v = v, possible_objects = possible_objects)

    num_frames = 2
    num_videos = 1
    fs = fill(num_frames, num_videos) #nummber of frames per video
    ps = fill(params, num_videos)

    videos = @trace(video_map(fs, ps), :videos)
end

export metacog
