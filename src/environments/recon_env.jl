module ReconEnv

using ..Interfaces
using Random, Statistics

export HolographicReconEnv

"""
    HolographicReconEnv(N, alpha, noise_level)

A high-tier environment for testing collective signaling under adversarial conditions.
Injects 'Holographic Decoys' (decoys that mimic the spectral signature of the real signal)
to confuse the population's synchronization.
"""
mutable struct HolographicReconEnv <: AbstractReflexiveEnv
    N::Int
    alpha::Float32
    noise_level::Float32
    state::Float32
    decoy_signal::Float32
    agent_states::Vector{Float32}
end

function HolographicReconEnv(N=1000, alpha=0.8, noise_level=0.2)
    return HolographicReconEnv(N, Float32(alpha), Float32(noise_level), 0.5f0, 0.0f0, rand(Float32, N))
end

function Interfaces.reset!(env::HolographicReconEnv)
    env.state = 0.5f0
    env.decoy_signal = 0.0f0
    env.agent_states = rand(Float32, env.N)
    return env.state
end

function Interfaces.step!(env::HolographicReconEnv, actions, collective_signal)
    # 1. Update Decoy (Adversarial shift)
    env.decoy_signal = env.state + 0.3f0 * sin(env.state * 10.0f0)
    
    # 2. Collective Signal Filtering
    coherence_error = abs(collective_signal - env.state)
    decoy_error = abs(collective_signal - env.decoy_signal)
    is_fooled = decoy_error < coherence_error
    
    # 3. State Evolution
    mean_action = mean(actions)
    pressure = env.alpha * (is_fooled ? env.decoy_signal : collective_signal)
    
    # Apply noise + base transition
    env.state = clamp(env.state + 0.1f0 * (mean_action - pressure) + env.noise_level * randn(Float32), 0.0f0, 1.0f0)
    
    # Update agent states
    env.agent_states .= clamp.(env.agent_states .+ 0.05f0 .* (env.state .- env.agent_states), 0.0f0, 1.0f0)
    
    return env.state
end

"""
    trigger_shock!(env; magnitude=2.0)

Simulates a massive informational shock (poisoning attack) by jumping 
the environment state and noise level.
"""
function trigger_shock!(env::HolographicReconEnv; magnitude=2.0f0)
    env.state = clamp(env.state + 0.5f0 * randn(Float32), 0.0f0, 1.0f0)
    env.noise_level *= magnitude
    println(">>> SHOCK INJECTED: Signal poisoned, Noise level doubled.")
end

function Interfaces.reward(env::HolographicReconEnv, state, actions)
    # Penalize both instability and being 'fooled' by the decoy
    return - (state - 0.5f0)^2 - 0.2f0 * abs(state - env.decoy_signal)
end

end # module
