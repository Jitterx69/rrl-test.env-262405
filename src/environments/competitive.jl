module Competitive

using ..Interfaces, ..Environments
import ..Interfaces: reset!, step!

export CompetitiveEnv

"""
    CompetitiveEnv(env)

A zero-sum wrapper for standard environments.
Two agents compete: 
- Agent A (Reflexive)
- Agent B (Adversarial/Noise)
"""
mutable struct CompetitiveEnv
    inner_env
    alpha_prime::Float32 # Inter-agent coupling factor
end

function CompetitiveEnv(env; alpha_prime=0.1f0)
    return CompetitiveEnv(env, Float32(alpha_prime))
end

function Interfaces.reset!(env::CompetitiveEnv)
    return reset!(env.inner_env)
end

function Interfaces.step!(env::CompetitiveEnv, action_a, action_b)
    net_action = action_a .- env.alpha_prime .* action_b
    # Pass net_action and treat action_b as the intercepted prediction (r_pred)
    return step!(env.inner_env, net_action, action_b)
end

function Interfaces.reward(env::CompetitiveEnv, state, action_a, action_b)
    net_action = action_a .- env.alpha_prime .* action_b
    r_base = reward(env.inner_env, state, net_action)
    return r_base, -r_base
end

export ElectronicWarfareEnv

"""
    ElectronicWarfareEnv(env)

An advanced competitive environment where the coupling factor `alpha` 
can be dynamically targeted or "jammed" by adversarial signals.
"""
mutable struct ElectronicWarfareEnv
    inner_env
    alpha_base::Float32
    jamming_factor::Float32
end

function ElectronicWarfareEnv(env; alpha_base=0.5f0)
    return ElectronicWarfareEnv(env, Float32(alpha_base), 0.0f0)
end

function Interfaces.reset!(env::ElectronicWarfareEnv)
    env.jamming_factor = 0.0f0
    return reset!(env.inner_env)
end

function Interfaces.step!(env::ElectronicWarfareEnv, action_a, action_b, jamming_signal=0.0f0)
    # Jamming reduces the coupling alpha for Agent A
    env.jamming_factor = clamp(Float32(jamming_signal), 0.0f0, 0.9f0)
    effective_alpha = env.alpha_base * (1.0f0 - env.jamming_factor)
    
    # Temporary alpha swap for inner_env
    old_alpha = env.inner_env.alpha
    env.inner_env.alpha = effective_alpha
    
    s = step!(env.inner_env, action_a, action_b)
    
    env.inner_env.alpha = old_alpha
    return s
end

function Interfaces.reward(env::ElectronicWarfareEnv, state, action_a, action_b)
    r_base = reward(env.inner_env, state, action_a) # Agent A goal
    # Agent B gets rewarded for Agent A's failure and its own jamming efficiency
    return r_base, -r_base + env.jamming_factor * 0.1f0
end

end # module
