"""
return rfs_vec, where the vector is the rfs indexed by receptive field
"""
function get_rfs_vec(real_objects::Vector{Detection2D},
                    params::Video_Params, v::Matrix{Real})
    #just one rf
    rec_field = Receptive_Field(p1 = (0, 0), p2 = (params.image_dim_x, params.image_dim_y)) #should make it from params
    real_rf = filter(p -> within(p, rec_field), real_objects)
    rfs_vec = get_rfs(rec_field, real_rf, params, v)
    #println("rfs_vec ", rfs_vec)
end

"""
returns rfs elements for a particular receptive field
"""
function get_rfs(rec_field::Receptive_Field, real::Vector{Detection2D}, params::Video_Params, v::Matrix{Real})

    #########################################################################
    #for imaginary objects
    lambda_fas = v[:,1]
    #fases = fill(fas, length(imaginary))
    #imagined_objects_2D = map(to_element, imaginary, fases)
    #imaginary = hallucination_distribution(params) #do I want to trace this? probably
    imagined_objects_2D = PoissonElement{Detection2D}(sum(v[:,1]), hallucination_distribution, (params, v, rec_field))

    #for real objects
    misses = v[:,2]
    #hitses = fill(hits, length(real))
    real_objects_2D = to_elements_real(real, misses)


    #println("imagined_objects_2D ", imagined_objects_2D)
    #println("real_objects_2D ", real_objects_2D)
    rfs = [imagined_objects_2D; real_objects_2D]

    rfs = RFSElements{Detection2D}(rfs)
    return rfs
end

"""
objects is a list of Detection2Ds. ps is a vector of probabilities indexed by object category
"""
function to_elements_real(objects::Vector{Detection2D}, detection_rates::Vector{Real})
    #will probably need to redo this stuff
    sd_x = 40. #might work????
    sd_y = 40.
    cov = [sd_x 0.; 0. sd_y;]

    n = length(objects)
    objects_2D = Vector{GeometricElement{Detection2D}}(undef, n) #no idea if that will work
    for i = 1:n
        x = objects[i][1]#trying to get x from Detection2D Tuple
        y = objects[i][2]
        cat = objects[i][3]
        detection_rate = detection_rates[cat] #get the detection rate for this category
        objects_2D[i] = GeometricElement{Detection2D}(detection_rate, object_distribution_image, ([x, y], cov, cat))
    end
    return objects_2D
end

"""
returns true if the object position is within rf
"""
function within(point::Detection2D, rf::Receptive_Field)
    x = point[1]
    y = point[2]
    x >= rf.p1[1] && x <= rf.p2[1] && y >= rf.p1[2] && y <= rf.p2[2] #top and right size will not be represented in rfs. nead it to be this way so intersections in receptive fields have reasonable explanations
end

function within(point::Tuple{Float64, Float64}, rf::Receptive_Field)
    x = point[1]
    y = point[2]
    x >= rf.p1[1] && x <= rf.p2[1] && y >= rf.p1[2] && y <= rf.p2[2] #top and right size will not be represented in rfs. nead it to be this way so intersections in receptive fields have reasonable explanations
end

export within

################################################################################
"""
set up receptive_fields
"""
function make_receptive_fields(params::Video_Params)

    # #square receptive fields. hardcoded for the 240 x 320 image
    # pixels = 80
    #
	# #layer 1 of receptive fields. 3x4, inside the image
    # n_horizontal = 4
    # n_vertical = 3
    # n = n_horizontal*n_vertical
    # receptive_fields_layer_1 = Vector{Receptive_Field}(undef, n) #of length n
    # for h = 1:n_horizontal
    #     for v = 1:n_vertical
    #         receptive_fields_layer_1[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1)*pixels, (v-1)*pixels), p2 = (h*pixels, v*pixels))
    #     end
    # end
    #
	# #layer 2 of receptive fields. 4x5 overlay
	# n_horizontal = 5
    # n_vertical = 4
    # n = n_horizontal*n_vertical
    # receptive_fields_layer_2 = Vector{Receptive_Field}(undef, n) #of length n
    # for h = 1:n_horizontal
    #     for v = 1:n_vertical
    #         receptive_fields_layer_2[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1.5)*pixels, (v-1.5)*pixels), p2 = ((h-0.5)*pixels, (v-0.5)*pixels))
    #     end
    # end
    #
	# #layer 3 of receptive fields. 3x5 overlay, tiled horizontally
	# n_horizontal = 5
    # n_vertical = 3
    # n = n_horizontal*n_vertical
    # receptive_fields_layer_3 = Vector{Receptive_Field}(undef, n) #of length n
    # for h = 1:n_horizontal
    #     for v = 1:n_vertical
    #         receptive_fields_layer_3[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1.5)*pixels, (v-1)*pixels), p2 = ((h-0.5)*pixels, v*pixels))
    #     end
    # end
    #
	# #layer 4 of receptive fields. 4x4 overlay, tiled vertically
	# n_horizontal = 4
    # n_vertical = 4
    # n = n_horizontal*n_vertical
    # receptive_fields_layer_4 = Vector{Receptive_Field}(undef, n) #of length n
    # for h = 1:n_horizontal
    #     for v = 1:n_vertical
    #         receptive_fields_layer_4[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1)*pixels, (v-1.5)*pixels), p2 = (h*pixels, (v-0.5)*pixels))
    #     end
    # end
    #
	# receptive_fields = vcat(receptive_fields_layer_1, receptive_fields_layer_2, receptive_fields_layer_3, receptive_fields_layer_4)

    receptive_fields = Vector{Receptive_Field}(undef, 1)
    receptive_fields[1] = Receptive_Field(p1 = (0, 0), p2 = (params.image_dim_x, params.image_dim_y))

    return receptive_fields
end

export make_receptive_fields
