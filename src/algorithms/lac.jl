module LAC

using ..SymbolicDistiller
using ..Interfaces
using LinearAlgebra, Statistics, Flux, Optimisers, Zygote, Random

export LACAgent, update_lac!, redistill_lyapunov!

"""
    LACAgent(policy, oracle, V, λ_lyap)

Lyapunov Actor-Critic (LAC) Agent with support for Symbolic Adaptation (NSRR).
"""
mutable struct LACAgent <: AbstractReflexiveAgent
    policy::GaussianPolicy
    oracle::ReflexiveOracle
    opt_state
    V::Any # Can be QuadraticLyapunov or a Symbolic String
    λ_lyap::Float32 
    β_stab::Float32 
    epsilon::Float32 
end

# ... (LACAgent constructor and call function remain same)

"""
    update_lac!(agent, obs, actions, rewards, next_obs)
"""
function update_lac!(agent::LACAgent, obs, actions, rewards, next_obs)
    grads = Zygote.gradient(agent.policy, agent.oracle) do p, o
        r_p = o(obs)
        
        # Policy Gradient
        μ, σ = p(obs)
        log_prob = -0.5f0 .* sum(((actions .- μ) ./ σ).^2 .+ 2.0f0 .* log.(σ))
        reward_loss = -mean(log_prob .* rewards)
        
        # Lyapunov Violation (Dynamic Dispatch based on V type)
        violation = if agent.V isa String
            # Using Symbolic Evaluator from NSRR
            v_curr = evaluate_expression(agent.V, obs)
            v_next = evaluate_expression(agent.V, next_obs)
            max.(0.0f0, v_next .- (1.0f0 - agent.epsilon) .* v_curr)
        else
            LyapunovDrift(agent.V, obs, next_obs; epsilon=agent.epsilon)
        end
        
        stability_loss = agent.λ_lyap * mean(violation)
        reg_loss = agent.β_stab * mean((next_obs .- obs).^2)
        
        return reward_loss + stability_loss + reg_loss
    end
    
    # ... (Optimization step logic same as before)
    # [Updating agent.opt_state, policy, oracle]
    
    # Dual Update
    violation_val = if agent.V isa String
        v_curr = evaluate_expression(agent.V, obs)
        v_next = evaluate_expression(agent.V, next_obs)
        mean(max.(0.0f0, v_next .- (1.0f0 - agent.epsilon) .* v_curr))
    else
        mean(LyapunovDrift(agent.V, obs, next_obs; epsilon=agent.epsilon))
    end
    agent.λ_lyap = clamp(agent.λ_lyap + 0.1f0 * violation_val, 1.0f0, 100.0f0)
    
    return violation_val
end

"""
    redistill_lyapunov!(agent, history)

Uses NSRR to extract a more accurate symbolic Lyapunov function from the 
agent's oracle performance history.
"""
function redistill_lyapunov!(agent::LACAgent, states)
    println(">>> NSRR: Re-distilling Symbolic Lyapunov for LAC...")
    # Fits a symbolic expression to the current oracle's representation
    agent.V = distill_lyapunov(agent.oracle, states)
    println(">>> LAC Updated with Analytical Proof: ", agent.V)
end

end # module
