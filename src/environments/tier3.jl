module Tier3

using ..Interfaces
using Random, LinearAlgebra

export Tier3Env

mutable struct Tier3Env <: AbstractReflexiveEnv
    alpha::Float32
    noise_std::Float32
    state::Vector{Float32}
    target::Vector{Float32}
end

function Tier3Env(alpha=0.1, noise_std=0.5)
    Tier3Env(Float32(alpha), Float32(noise_std), [0.0f0, 0.0f0], [1.0f0, 1.0f0])
end

function Interfaces.reset!(env::Tier3Env)
    env.state = [0.0f0, 0.0f0]
    return env.state
end

function Interfaces.step!(env::Tier3Env, action, r_pred)
    noise = env.noise_std .* randn(Float32, 2)
    new_s = env.state .+ Float32(action) .- env.alpha .* Float32(r_pred) .+ noise
    env.state = clamp.(new_s, -100.0f0, 100.0f0) # Stability Guard
    return env.state
end

function Interfaces.reward(env::Tier3Env, state, action)
    return -sum((state .- env.target).^2) - 0.1f0 * sum(action.^2)
end

end # module
