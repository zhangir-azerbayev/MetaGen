module MetaGen

using Gen
using GenRFS
using Distributions

function __init__()
    @load_generated_functions
end

include("declaring_structs.jl")
include("custom_distributions.jl")
include("geometry_optics.jl")
include("inverse_optics.jl")
include("receptive_fields.jl")
include("video.jl")
include("involution.jl")
include("pf_inference.jl")
include("metacog.jl")
include("printing.jl")

end # module
