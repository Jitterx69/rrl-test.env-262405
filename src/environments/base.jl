module Environments

using ..Interfaces

# This file now acts as a coordinator, including individual tier modules
include("tier1.jl")
using .Tier1
include("tier2.jl")
using .Tier2
include("tier3.jl")
using .Tier3

export Tier1Env, Tier2Env, Tier3Env

end # module
