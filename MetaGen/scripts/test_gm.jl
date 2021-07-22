using Revise
using MetaGen
using Gen
#using StatProfilerHTML

#call it
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
num_frames = 1
num_videos = 1
params = Video_Params()
gt_trace,_ = Gen.generate(main, (num_videos, num_frames, params));
#println(gt_trace)
gt_choices = get_choices(gt_trace)
