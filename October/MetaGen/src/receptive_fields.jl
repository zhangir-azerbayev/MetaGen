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
    lambda_fas = params.v[:,1]
    #fases = fill(fas, length(imaginary))
    #imagined_objects_2D = map(to_element, imaginary, fases)
    #imaginary = hallucination_distribution(params) #do I want to trace this? probably
    imagined_objects_2D = PoissonElement{Detection2D}(sum(params.v[:,1]), hallucination_distribution, (params, rec_field))

    #for real objects
    misses = params.v[:,2]
    #hitses = fill(hits, length(real))
    real_objects_2D = to_elements_real(real, misses)

    rfs = vcat(imagined_objects_2D, real_objects_2D)

    rfs = RFSElements{Detection2D}(rfs)
    return rfs
end

#objects is a list of Detection2Ds. ps is a vector of probabilities indexed by object category
function to_elements_real(objects::Vector{Detection2D}, misses::Vector{Float64})
    #will probably need to redo this stuff
    sd_x = 10. #might work????
    sd_y = 10.
    cov = [sd_x 0.; 0. sd_y;]

    n = length(objects)
    objects_2D = Vector{BernoulliElement{Detection2D}}(undef, n) #no idea if that will work
    for i = 1:n
        x = objects[n][1]#trying to get x from Detection2D Tuple
        y = objects[n][2]
        cat = objects[n][3]
        miss = misses[cat] #get the fa for this category
        objects_2D[i] = BernoulliElement{Detection2D}(1-miss, object_distribution_image, ([x, y], cov, cat))
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
