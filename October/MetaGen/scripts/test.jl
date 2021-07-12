using MetaGen
using Gen

num_videos = 2
num_frames = 2

objects_observed = Matrix{Array{Detection2D}}(undef, num_videos, num_frames)
camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)
for v=1:num_videos
	f = 1

	labels = [2, 5]
	xs = [50.1, 225.5]
	ys = [60.2, 173.1]
	temp = Array{Detection2D}(undef, length(labels))
	for i = 1:length(labels)
		label = labels[i]
		x = xs[i]
		y = ys[i]
		temp[i] = (x, y, label) #so now I have an array of detections
	end
	#turn that array of detections into an array of an array of detections sorted by receptive_field
	#temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
	objects_observed[v, f] = temp

	c = Camera_Params(camera_location = Coordinate(0.01,0.02,0.001), camera_focus = Coordinate(1.0,1.0,1.0))
	camera_trajectories[v, f] = c

	#################################
	#frame 2
	for f = 2:2

		labels = [2]
		xs = [50.1]
		ys = [60.2]
		temp = Array{Detection2D}(undef, length(labels))
		for i = 1:length(labels)
			label = labels[i]
			x = xs[i]
			y = ys[i]
			temp[i] = (x, y, label) #so now I have an array of detections
		end
		#turn that array of detections into an array of an array of detections sorted by receptive_field
		#temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
		#objects_observed[v, f] = temp_sorted_into_rfs
		objects_observed[v, f] = temp

		c = Camera_Params(camera_location = Coordinate(0.01,0.02,0.001), camera_focus = Coordinate(1.0,1.0,1.0))
		camera_trajectories[v, f] = c
	end
end

cm = choicemap()
for v = 1:num_videos
	# cm[:videos => v => :v_matrix => :lambda_fa => 1 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 2 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 3 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 4 => :fa] = 0.000001
	# cm[:videos => v => :v_matrix => :lambda_fa => 5 => :fa] = 0.1 #super high false alarm rate for category 5
	# cm[:videos => v => :v_matrix => :miss_rate => 1 => :miss] = 0.000001 #tiny miss rate for category 1
	# cm[:videos => v => :v_matrix => :miss_rate => 2 => :miss] = 0.01
	# cm[:videos => v => :v_matrix => :miss_rate => 3 => :miss] = 0.01
	# cm[:videos => v => :v_matrix => :miss_rate => 4 => :miss] = 0.99 #huge miss rate for category 4
	# cm[:videos => v => :v_matrix => :miss_rate => 5 => :miss] = 0.01
	for f = 1:num_frames
		camera_params = camera_trajectories[v, f]
		cm[:videos => v => :frame_chain => f => :camera => :camera_location_x] = camera_params.camera_location.x
		cm[:videos => v => :frame_chain => f => :camera => :camera_location_y] = camera_params.camera_location.y
		cm[:videos => v => :frame_chain => f => :camera => :camera_location_z] = camera_params.camera_location.z
		cm[:videos => v => :frame_chain => f => :camera => :camera_focus_x] = camera_params.camera_focus.x
		cm[:videos => v => :frame_chain => f => :camera => :camera_focus_y] = camera_params.camera_focus.y
		cm[:videos => v => :frame_chain => f => :camera => :camera_focus_z] = camera_params.camera_focus.z

		#for rf = 1:num_receptive_fields
			#println("objects_observed[v, f][rf] ", objects_observed[v, f][rf])
			#println("type ", typeof(objects_observed[v, f][rf]))
		cm[:videos => v => :frame_chain => f => :observations_2D] = convert(Array{Any, 1}, objects_observed[v, f])
		#end
	end
	#def
end

gt_trace,_ = Gen.generate(main, (num_videos, num_frames), cm);
#println(gt_trace)
gt_choices = get_choices(gt_trace)

regenerate(gt_trace, select(:videos => 1 => :init_scene));
regenerate(gt_trace, select(:videos => 2 => :v_matrix => :lambda_fa => 1 => :fa));
