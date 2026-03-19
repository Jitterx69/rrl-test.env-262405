module PopulationEnv

using ..Interfaces
using Random, Statistics

export MarketPopulationEnv

"""
    MarketPopulationEnv(N, alpha)

A massive multi-agent environment where the global state evolves 
based on the coherent mean of the reflexive population.
"""
mutable struct MarketPopulationEnv <: AbstractReflexiveEnv
    N::Int
    alpha::Float32
    state::Float32
    agent_states::Vector{Float32}
end

function MarketPopulationEnv(N=1000, alpha=0.5)
    MarketPopulationEnv(N, Float32(alpha), 0.5f0, rand(Float32, N))
end

function Interfaces.reset!(env::MarketPopulationEnv)
    env.state = 0.5f0
    env.agent_states = rand(Float32, env.N)
    return env.state
end

function Interfaces.step!(env::MarketPopulationEnv, actions, collective_signal)
    # actions: N-vector of agent decisions
    # collective_signal: scalar "shared belief"
    
    # Global state driven by collective action + reflexive coupling
    mean_action = mean(actions)
    
    # Coherent pressure term (Super-radiance effect)
    pressure = env.alpha * collective_signal
    
    env.state = clamp(env.state + 0.1f0 * (mean_action - pressure), 0.0f0, 1.0f0)
    
    # Micro-updates for agents based on global state
    env.agent_states .= clamp.(env.agent_states .+ 0.05f0 .* (env.state .- env.agent_states), 0.0f0, 1.0f0)
    
    return env.state
end

function Interfaces.reward(env::MarketPopulationEnv, state, actions)
    # Group reward for collective stability
    return - (state - 0.5f0)^2 - 0.1f0 * var(actions)
end

end # module
