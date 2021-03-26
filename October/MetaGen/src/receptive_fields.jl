#return rfs_vec, where the vector is the rfs indexed by receptive field
function get_rfs_vec(rec_fields::Vector{Receptive_Field},
                            real_objects::Vector{Detection2D},
                            params::Video_Params)

    real_rf = map(rf -> filter(p -> within(p, rf), real_objects), rec_fields)
    paramses = fill(params, length(rec_fields))
    rfs_vec = map(get_rfs, rec_fields, real_rf, paramses)
end

#rerutns rfs elements for a particular receptive field
function get_rfs(rec_field::Receptive_Field, real::Vector{Detection2D}, params::Video_Params)

    #########################################################################
    #for imaginary objects
    lambda_fas = params.v[:,1] ./ params.num_receptive_fields

    #for real objects
    lambda_hits = params.v[:,2]
    #hitses = fill(hits, length(real))
    #real_objects_2D = map(to_element, imaginary, 1 .- hitses)
    reals = to_elements(real, lambda_hits)

    imaginaries = to_elements_imaginary(rec_field, lambda_fas)

    rfs = vcat(reals, imaginaries)
    rfs = RFSElements{Detection2D}(rfs)
end

#objects is a list of Detection2Ds. ps is a vector of lambdas indexed by object category
function to_elements(objects::Vector{Detection2D}, lambda_hits::Vector{Float64})
    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.; 0. sd_y;]

    n = length(objects)
    objects_2D = Vector{PoissonElement{Detection2D}}(undef, n)
    for i = 1:n
        x = objects[n][1]#trying to get x from Detection2D Tuple
        y = objects[n][2]
        cat = objects[n][3]
        lambda_hit = lambda_hits[cat] #get the fa for this category
        objects_2D[i] = PoissonElement{Detection2D}(lambda_hit, object_distribution_image, ([x, y], cov, cat))
    end
    return objects_2D
end

#lambda_fas must have already been adjusted for the number of receptive fields
function to_elements_imaginary(rec_field::Receptive_Field, lambda_fas::Vector{Float64})
    sd_x = 1.
    sd_y = 1.
    cov = [sd_x 0.; 0. sd_y;]

    n = length(lambda_fas)
    imaginary_objects_2D = Vector{PoissonElement{Detection2D}}(undef, n)
    for cat = 1:n
        x = uniform(rec_field.p1[1], rec_field.p2[1])#sample uniformly from this receptive field
        y = uniform(rec_field.p1[2], rec_field.p2[2])
        lambda_fa = lambda_fas[cat] #get the fa for this category
        imaginary_objects_2D[cat] = PoissonElement{Detection2D}(lambda_fa, object_distribution_image, ([x, y], cov, cat))
    end
    return imaginary_objects_2D
end

# returns true if the object position is within rf
function within(point::Detection2D, rf::Receptive_Field)
    x = point[1]
    y = point[2]
    x > rf.p1[1] && x < rf.p2[1] && y > rf.p1[2] && y < rf.p2[2]
end
