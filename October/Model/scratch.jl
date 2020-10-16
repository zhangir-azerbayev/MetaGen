#Scratch file





include("gm.jl")

##For Simulated data
n_percepts = 2
n_frames = [1,3]
possible_objects = ["person","bicycle","car","motorcycle","airplane"]

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
gt_trace,_ = Gen.generate(gm, (possible_objects, n_percepts, n_frames))
gt_reality,gt_V,gt_percepts = Gen.get_retval(gt_trace)
gt_choices = get_choices(gt_trace)

println("gt_choices")
display(gt_choices)
