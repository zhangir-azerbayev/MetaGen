module MetaGen

using Gen
using GenRFS
using Distributions
using Combinatorics
using Random
#using Plots
using PyPlot
const plt = PyPlot
#import PyPlot; const plt = PyPlot

include("declaring_structs.jl")
include("custom_distributions.jl")
include("geometry_optics.jl")
include("inverse_optics.jl")
include("receptive_fields.jl")
include("video.jl")
include("involution.jl")
include("particle_filter_helper_functions.jl")
include("pf_inference.jl")
include("main.jl")
include("printing.jl")
include("visualizations.jl")

function __init__()
    @load_generated_functions
end

end # module
