module Interfaces

export AbstractReflexiveEnv, AbstractReflexiveAgent
export reset!, step!, reward, observe

"""
    AbstractReflexiveEnv
Base type for all reflexive environments (\$T, R, \\pi\$).
"""
abstract type AbstractReflexiveEnv end

"""
    AbstractReflexiveAgent
Base type for all reflexive algorithms.
"""
abstract type AbstractReflexiveAgent end

# Mandatory methods for environments
function reset! end
function step! end
function reward end
function observe end

# Mandatory methods for agents
function update! end

end # module
