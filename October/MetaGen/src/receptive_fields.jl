#return rfs_vec, where the vector is the rfs indexed by receptive field
function get_rfs_vec(rec_fields::Vector{Receptive_Field},
                            imaginary_objects::Vector{Detection2D},
                            real_objects::Vector{Detection2D},
                            params::Video_Params)

    imaginary_rf = map(rf -> filter(p -> within(p, rf), imaginary_objects), rec_fields)
    real_rf = map(rf -> filter(p -> within(p, rf), real_objects), rec_fields)
    paramses = fill(params, length(rec_fields))
    rfs_vec = map(get_rfs, rec_fields, imaginary_rf, real_rf, paramses)
end

#rerutns rfs elements for a particular receptive field
function get_rfs(rec_field::Receptive_Field, imaginary::Vector{Detection2D}, real::Vector{Detection2D}, params::Video_Params)

    #########################################################################
    #for imaginary objects
    lambda_fas = params.v[:,1]
    #fases = fill(fas, length(imaginary))
    #imagined_objects_2D = map(to_element, imaginary, fases)
    imagined_objects_2D = to_elements_imaginary(imaginary) #don't need lambda_fas because those went into making possible hallucinations at the image level

    #for real objects
    lambda_hits = params.v[:,2]
    #hitses = fill(hits, length(real))
    real_objects_2D = to_elements_real(real, lambda_hits)

    rfs = vcat(imagined_objects_2D, real_objects_2D)

    #if vector was empty, put in a BernoulliElement with p=0
    if length(rfs)==0
        sd_x = 1.
        sd_y = 1.
        cov = [sd_x 0.; 0. sd_y;]
        rfs = Vector{BernoulliElement{Detection2D}}(undef, 1)
        rfs[1] = BernoulliElement{Detection2D}(0.0, object_distribution_image, ([-1000., -1000.], cov, 0))
    end

    rfs = RFSElements{Detection2D}(rfs)
    return rfs
end

#objects is a list of Detection2Ds. ps is a vector of probabilities indexed by object category
function to_elements_real(objects::Vector{Detection2D}, lambdas::Vector{Float64})
    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.; 0. sd_y;]

    n = length(objects)
    objects_2D = Vector{PoissonElement{Detection2D}}(undef, n) #no idea if that will work
    for i = 1:n
        x = objects[n][1]#trying to get x from Detection2D Tuple
        y = objects[n][2]
        cat = objects[n][3]
        lambda = lambdas[cat] #get the fa for this category
        objects_2D[i] = PoissonElement{Detection2D}(lambda, object_distribution_image, ([x, y], cov, cat))
    end
    return objects_2D
end

function to_elements_imaginary(objects::Vector{Detection2D})
    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.; 0. sd_y;]

    n = length(objects)
    objects_2D = Vector{BernoulliElement{Detection2D}}(undef, n) #no idea if that will work
    for i = 1:n
        x = objects[n][1]#trying to get x from Detection2D Tuple
        y = objects[n][2]
        cat = objects[n][3]
        objects_2D[i] = BernoulliElement{Detection2D}(1.0, object_distribution_image, ([x, y], cov, cat))
    end
    return objects_2D
end

# returns true if the object position is within rf
function within(point::Detection2D, rf::Receptive_Field)
    x = point[1]
    y = point[2]
    x > rf.p1[1] && x < rf.p2[1] && y > rf.p1[2] && y < rf.p2[2]
end

export within
