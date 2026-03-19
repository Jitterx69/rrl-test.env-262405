module Tier1

using ..Interfaces
using Random

export Tier1Env

mutable struct Tier1Env <: AbstractReflexiveEnv
    alpha::Float32
    noise_std::Float32
    state::Float32
    target::Float32
end

function Tier1Env(alpha=1.0, noise_std=0.05)
    Tier1Env(Float32(alpha), Float32(noise_std), 0.0f0, 0.0f0)
end

function Interfaces.reset!(env::Tier1Env)
    env.state = 0.0f0
    return env.state
end

function Interfaces.step!(env::Tier1Env, action, r_pred)
    # Handle both scalar and vector inputs for robustness
    a_val = action isa AbstractVector ? Float32(action[1]) : Float32(action)
    rp_val = r_pred isa AbstractVector ? Float32(r_pred[1]) : Float32(r_pred)
    
    noise = env.noise_std * randn(Float32)
    next_s = env.state + a_val - env.alpha * rp_val + noise
    env.state = clamp(next_s, -10.0f0, 10.0f0) 
    return env.state
end

function Interfaces.reward(env::Tier1Env, state, action)
    a_val = action isa AbstractVector ? Float32(action[1]) : Float32(action)
    return - sum((state .- env.target).^2) - 0.1f0 * sum(a_val.^2)
end

end # module
