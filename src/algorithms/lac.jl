module LAC

using Flux, Optimisers, Zygote, Statistics
using ..Interfaces
using ..Architectures
using ..MathUtils
using ..StabilityUtils

export LACAgent, update_lac!

"""
    LACAgent(policy, oracle, V, λ_lyap)

Lyapunov Actor-Critic (LAC) Agent.
Uses a Lyapunov candidate function V to enforce global asymptotic stability 
within the reflexive training loop.
"""
mutable struct LACAgent <: AbstractReflexiveAgent
    policy::GaussianPolicy
    oracle::ReflexiveOracle
    opt_state
    V::QuadraticLyapunov
    λ_lyap::Float32 # Lagrangian multiplier for stability constraint
    β_stab::Float32 # Fixed stability regularizer
    epsilon::Float32 # Lyapunov decay rate
end

function LACAgent(in_dim::Int, action_dim::Int, hidden_dim::Int=64; 
                  lr=1f-3, β_stab=0.1f0, epsilon=0.01f0)
    policy = GaussianPolicy(in_dim, action_dim)
    oracle = ReflexiveOracle(in_dim, 1) # Single scalar prediction for stability
    # Quadratic candidate based on state dimension
    V = QuadraticLyapunov(in_dim)
    
    opt_p = Optimisers.setup(Optimisers.Adam(lr), policy)
    opt_o = Optimisers.setup(Optimisers.Adam(lr), oracle)
    
    return LACAgent(policy, oracle, (opt_p, opt_o), V, 1.0f0, β_stab, epsilon)
end

function (a::LACAgent)(obs)
    μ, σ = a.policy(obs)
    action = μ .+ σ .* randn(Float32, size(μ))
    r_p = a.oracle(obs)
    return action, r_p
end

"""
    update_lac!(agent, obs, actions, rewards, next_obs)

Update the LACAgent with Lyapunov-constrained optimization.
Objective: Maximize Reward subject to ΔV ≤ -εV.
"""
function update_lac!(agent::LACAgent, obs, actions, rewards, next_obs)
    # 1. Compute Gradients
    grads = Zygote.gradient(agent.policy, agent.oracle) do p, o
        # Oracle prediction
        r_p = o(obs)
        
        # Reward Loss (Policy Gradient)
        μ, σ = p(obs)
        log_prob = -0.5f0 .* sum(((actions .- μ) ./ σ).^2 .+ 2.0f0 .* log.(σ))
        reward_loss = -mean(log_prob .* rewards)
        
        # Lyapunov Violation: max(0, V(next_s) - (1-ε)V(s))
        violation = LyapunovDrift(agent.V, obs, next_obs; epsilon=agent.epsilon)
        
        # Constrained Objective: L = RewardLoss + λ * Violation
        # Note: In a full LAC, λ is updated via dual descent.
        # For this prototype, we use a high-penalty Lagrangian.
        stability_loss = agent.λ_lyap * mean(violation)
        
        # Fixed regularizer for smooth convergence
        reg_loss = agent.β_stab * mean((next_obs .- obs).^2)
        
        return reward_loss + stability_loss + reg_loss
    end
    
    # Update weights
    new_opt_p = agent.opt_state[1]
    new_p = agent.policy
    if grads[1] !== nothing
        new_opt_p, new_p = Optimisers.update!(agent.opt_state[1], agent.policy, grads[1])
    end
    
    new_opt_o = agent.opt_state[2]
    new_o = agent.oracle
    if grads[2] !== nothing
        new_opt_o, new_o = Optimisers.update!(agent.opt_state[2], agent.oracle, grads[2])
    end
    
    agent.opt_state = (new_opt_p, new_opt_o)
    agent.policy = new_p
    agent.oracle = new_o
    
    # 2. Dual Update (Update λ_lyap based on violation)
    # Simple dual heuristic: if violating, increase penalty
    violation_val = mean(LyapunovDrift(agent.V, obs, next_obs; epsilon=agent.epsilon))
    agent.λ_lyap = clamp(agent.λ_lyap + 0.1f0 * violation_val, 1.0f0, 100.0f0)
    
    return violation_val
end

end # module
