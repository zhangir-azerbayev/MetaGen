using Revise
using MetaGen
using Gen
using StatProfilerHTML

#call it
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
num_frames = 2
num_videos = 3
gt_trace,_ = Gen.generate(main, (num_videos, num_frames))
#println(gt_trace)
gt_choices = get_choices(gt_trace)
