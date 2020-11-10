using Revise
using MetaGen
using Gen
using StatProfilerHTML

#possible_objects = ["person","bicycle","car","motorcycle","airplane"]
possible_objects = [1, 2, 3, 4, 5]
#call it
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#println(gt_trace)
gt_choices = get_choices(gt_trace)
