using Gen

include("mbrfs_simplified.jl")
include("shared_functions.jl")

FA = 0.5
M = 0.2
E = 0.9

mu = Vector{Float64}(undef, 2)
cov_mat = Matrix{Float64}(undef, 2, 2)

mu[1] = -5
mu[2] = -2

cov_mat[1,1] = 0.1
cov_mat[2,2] = 0.1
cov_mat[1,2] = 0
cov_mat[2,1] = 0

low_x = 20.0
low_y = 40.0
high_x = 20.1
high_y = 40.1

# params = MBRFSParams([FA, (1-M)*E],
#                          [mvuniform, mvnormal],
#                          [(low_x,low_y, high_x, high_y), (mu,cov_mat)])

# params = MBRFSParams([FA, FA],
#                          [mvnormal, mvnormal],
#                          [(mu,cov_mat), (mu,cov_mat)])

# params = MBRFSParams([FA, FA],
#                     [mvuniform, mvuniform],
#                     [(low_x,low_y, high_x, high_y), (low_x,low_y, high_x, high_y)])

# xs = mbrfs(params)
# println("xs", xs)
#
# Gen.logpdf(mbrfs, xs, params)

# params = MBRFSParams([1, 1],
#                     [mvnormal, mvnormal],
#                     [(mu,cov_mat), (mu,cov_mat)])
#
# #xs = mbrfs(params)
# xs = [[-5.358463202632788, -1.5894397392473787],[-5.358463202632788, -1.5894397392473787]] #each had lpdf of -1.020569950170379 for p = 0.36
# #together, they have
# Gen.logpdf(mbrfs, xs, params)

#Test 1
a = Gen.logpdf(mvnormal, [-5, -2], mu, cov_mat) #0.4647

params = MBRFSParams([1],
                    [mvnormal],
                    [(mu,cov_mat)])
xs = [[-5, -2]]
b = Gen.logpdf(mbrfs, xs, params) #0.4647

println("a ", a)
println("b ", b)

println(a==b) #Checks out

#Test 2
a = log(0.5) + Gen.logpdf(mvnormal, [-5, -2], mu, cov_mat) # -0.693 + 0.4647

params = MBRFSParams([0.5],
                    [mvnormal],
                    [(mu,cov_mat)])
xs = [[-5, -2]]
b = Gen.logpdf(mbrfs, xs, params)

println("a ", a)
println("b ", b)

println(a==b) #Checks out

#Test 3. Up it to 2 rvs.
#log(2) is because we're dealing with sets where order doesn't matter, so that's for the combinatorics part
a = log(2) + Gen.logpdf(mvnormal, [-5.5, -2.5], mu, cov_mat) + Gen.logpdf(mvnormal, [-5, -2], mu, cov_mat)

params = MBRFSParams([1, 1],
                   [mvnormal, mvnormal],
                   [(mu,cov_mat), (mu,cov_mat)])
xs = [[-5, -2], [-5.5, -2.5]]
b = Gen.logpdf(mbrfs, xs, params)

println("a ", a)
println("b ", b)

println(a==b)

#Test 4. Up it to 2 rvs.
#no log(2) since only one data association is possible here
a = Gen.logpdf(mvuniform, [1.5, 3.5], 1, 3, 3, 5) + Gen.logpdf(mvnormal, [-5, -2], mu, cov_mat) #logpdf(mvuniform) is log(1/4)=-1.386

params = MBRFSParams([1, 1],
                   [mvuniform, mvnormal],
                   [(1, 3, 3, 5), (mu,cov_mat)])
xs = [[1.5, 3.5], [-5, -2]]
b = Gen.logpdf(mbrfs, xs, params)

println("a ", a)
println("b ", b)

println(a==b) #checks out! Hurray!!!!!!!!!!
