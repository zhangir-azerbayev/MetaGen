module MetaGen

using Gen
using GenRFS
using Distributions

function __init__()
    @load_generated_functions
end

include("custom_distributions.jl")
include("video.jl")
include("metacog.jl")

end # module
