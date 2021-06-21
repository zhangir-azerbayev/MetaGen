module MetaGen

using Gen
using GenRFS
using Distributions
using Plots
#import PyPlot; const plt = PyPlot

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
include("visualizations.jl")

end # module
