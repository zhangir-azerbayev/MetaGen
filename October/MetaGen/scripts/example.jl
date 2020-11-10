using MetaGen
using Gen

#possible_objects = ["person","bicycle","car","motorcycle","airplane"]
possible_objects = [1, 2, 3, 4, 5]
#call it
gt_trace,_ = Gen.generate(metacog, (possible_objects,))
gt_choices = get_choices(gt_trace)
