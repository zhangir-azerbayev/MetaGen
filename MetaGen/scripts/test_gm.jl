using Revise
using MetaGen
using Gen
using Profile
using StatProfilerHTML
using GenRFS

#Profile.init(; n = 10^4, delay = 1e-5)

#GenRFS.modify_partition_ctx!(1000)

#call it
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
num_frames = 300
num_videos = 100
params = Video_Params()

@time gt_trace,_ = Gen.generate(main, (false, num_videos, num_frames, params));

@profilehtml Gen.generate(main, (num_videos, num_frames, params));


#Profile.init(; n = 10^7, delay = 1e-6)
@profilehtml gt_trace,_ = Gen.generate(main, (num_videos, num_frames, params));
#println(gt_trace)
#gt_choices = get_choices(gt_trace)
