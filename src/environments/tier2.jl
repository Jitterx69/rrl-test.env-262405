module Tier2

using ..Interfaces
using Random

export Tier2Env

mutable struct Tier2Env <: AbstractReflexiveEnv
    alpha::Float32
    noise_std::Float32
    state::Float32
    target::Float32
end

function Tier2Env(alpha=5.0, noise_std=0.5)
    Tier2Env(Float32(alpha), Float32(noise_std), 0.0f0, 2.0f0)
end

function Interfaces.reset!(env::Tier2Env)
    env.state = 0.0f0
    return env.state
end

function Interfaces.step!(env::Tier2Env, action, r_pred)
    noise = env.noise_std * randn(Float32)
    new_s = env.state + tanh(Float32(action)) - env.alpha * Float32(r_pred) + noise
    env.state = clamp(new_s, -100.0f0, 100.0f0) # Stability Guard
    return env.state
end

function Interfaces.reward(env::Tier2Env, state, action)
    return - sum((state .- env.target).^2) - 0.05f0 * sum(action.^2)
end

end # module
