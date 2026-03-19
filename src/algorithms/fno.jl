module FNO

using Flux, Optimisers, Zygote, Statistics
using ..Interfaces
using ..Architectures
using ..MathUtils

export FNOAgent, update_fno!

"""
    FNOAgent(policy, oracle, opt_state)

A Neural Operator agent that utilizes Gated Spectral Units for reflexive prediction.
Optimizes for discretization-invariant stability.
"""
mutable struct FNOAgent <: AbstractReflexiveAgent
    policy::GaussianPolicy
    oracle::GatedSpectralOracle
    opt_state
    β_stab::Float32
    λ_spectral::Float32
end

function FNOAgent(in_dim::Int, action_dim::Int, hidden_dim::Int=64, modes::Int=16; 
                  lr=1f-3, β_stab=0.1f0, λ_spectral=0.01f0)
    policy = GaussianPolicy(in_dim, action_dim)
    oracle = GatedSpectralOracle(in_dim, 1, hidden_dim, modes)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), (policy, oracle))
    return FNOAgent(policy, oracle, opt_state, β_stab, λ_spectral)
end

function (a::FNOAgent)(obs)
    μ, σ = a.policy(obs)
    # Re-parameterization trick for stochastic actions
    action = μ .+ σ .* randn(Float32, size(μ))
    # Predict reflexive signal using the Gated Spectral Unit
    r_p = a.oracle(obs)
    return action, r_p
end

"""
    update_fno!(agent, batch)

Update the FNOAgent using a combination of Policy Gradient,
Stability Loss, and Spectral Regularization.
"""
function update_fno!(agent::FNOAgent, obs, actions, rewards, next_obs)
    # Functional gradient update
    grads = Zygote.gradient(agent.policy, agent.oracle) do p, o
        # 1. Oracle Prediction
        r_p = o(obs)
        
        # 2. Stability Loss (Fixed-point residual)
        # s_{t+1} = f(s_t, a_t, r_p) - we assume next_obs is the result
        stab_loss = sum((next_obs .- obs).^2)
        
        # 3. Reward Maximization (Simplified PG)
        # log π(a|s) * R
        μ, σ = p(obs)
        log_prob = -0.5f0 .* sum(((actions .- μ) ./ σ).^2 .+ 2.0f0 .* log.(σ))
        reward_loss = -mean(log_prob .* rewards)
        
        # 4. Spectral Regularization (Optional: penalize high-freq energy in oracle)
        # (This is conceptual for the prototype)
        
        return reward_loss + agent.β_stab * stab_loss
    end
    
    agent.opt_state, (agent.policy, agent.oracle) = Optimisers.update(agent.opt_state, (agent.policy, agent.oracle), grads[1], grads[2])
end

end # module
