module FNO

using ..TopologicalAnalysis
using ..Interfaces
using LinearAlgebra, Statistics, Flux, Optimisers, Zygote, Random

export FNOAgent, update_fno!

"""
    FNOAgent(policy, oracle, opt_state)
"""
mutable struct FNOAgent <: AbstractReflexiveAgent
    policy::GaussianPolicy
    oracle::GatedSpectralOracle
    opt_state
    β_stab::Float32
    λ_spectral::Float32
    λ_tib::Float32 # Topological Information Bottleneck multiplier
end

function FNOAgent(in_dim::Int, action_dim::Int, hidden_dim::Int=64, modes::Int=16; 
                  lr=1f-3, β_stab=0.1f0, λ_spectral=0.01f0, λ_tib=0.05f0)
    policy = GaussianPolicy(in_dim, action_dim)
    oracle = GatedSpectralOracle(in_dim, 1, hidden_dim, modes)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), (policy, oracle))
    return FNOAgent(policy, oracle, opt_state, β_stab, λ_spectral, λ_tib)
end

# (call function remains same)

"""
    update_fno!(agent, obs, actions, rewards, next_obs, obs_history)
"""
function update_fno!(agent::FNOAgent, obs, actions, rewards, next_obs, obs_history=[])
    # Functional gradient update
    grads = Zygote.gradient(agent.policy, agent.oracle) do p, o
        r_p = o(obs)
        
        # 1. Stability Loss
        stab_loss = sum((next_obs .- obs).^2)
        
        # 2. Topological Information Bottleneck (TIB)
        # Minimize the 'pressure' (loops/chaos) of the recent history
        tib_loss = if !isempty(obs_history)
            topological_loss(obs_history; lambda=agent.λ_tib)
        else
            0.0f0
        end
        
        # 3. Reward Maximization
        μ, σ = p(obs)
        log_prob = -0.5f0 .* sum(((actions .- μ) ./ σ).^2 .+ 2.0f0 .* log.(σ))
        reward_loss = -mean(log_prob .* rewards)
        
        return reward_loss + agent.β_stab * stab_loss + tib_loss
    end
    
    agent.opt_state, (agent.policy, agent.oracle) = Optimisers.update(agent.opt_state, (agent.policy, agent.oracle), grads[1], grads[2])
end

end # module
