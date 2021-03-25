using Revise
using MetaGen
using Gen
using StatProfilerHTML

#possible_objects = ["person","bicycle","car","motorcycle","airplane"]
possible_objects = [1, 2, 3, 4, 5]
#call it
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
num_frames = 3
num_videos = 2
gt_trace,_ = Gen.generate(metacog, (possible_objects, num_videos, num_frames))
#println(gt_trace)
gt_choices = get_choices(gt_trace)

#################################################################################
num_particles = 100
#tr = unfold_particle_filter(num_particles);
