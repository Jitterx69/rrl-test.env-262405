module NeuralLAC

using Flux, Optimisers, Zygote, Statistics
using ..Interfaces
using ..Architectures
using ..MathUtils
using ..StabilityUtils

export NeuralLACAgent, update_neural_lac!

"""
    NeuralLACAgent(policy, oracle, V_net)

Advanced Lyapunov Actor-Critic that *learns* the Lyapunov manifold.
Combines spectral dynamics with formal safety.
"""
mutable struct NeuralLACAgent <: AbstractReflexiveAgent
    policy::GaussianPolicy
    oracle::SpectralOracle
    V_net::NeuralLyapunov
    opt_state
    λ_lyap::Float32
    beta_stab::Float32
end

function NeuralLACAgent(in_dim::Int, action_dim::Int; hidden_dim=64, modes=16, lr=1f-3)
    policy = GaussianPolicy(in_dim, action_dim)
    oracle = SpectralOracle(in_dim, 1, hidden_dim, modes)
    V_net  = NeuralLyapunov(in_dim, hidden_dim)
    
    opt_p = Optimisers.setup(Optimisers.Adam(lr), policy)
    opt_o = Optimisers.setup(Optimisers.Adam(lr), oracle)
    opt_v = Optimisers.setup(Optimisers.Adam(lr), V_net)
    
    return NeuralLACAgent(policy, oracle, V_net, (opt_p, opt_o, opt_v), 1.0f0, 0.1f0)
end

function (a::NeuralLACAgent)(obs)
    μ, σ = a.policy(obs)
    action = μ .+ σ .* randn(Float32, size(μ))
    r_p = a.oracle(obs)
    return action, r_p
end

"""
    update_neural_lac!(agent, obs, actions, rewards, next_obs)

Learns the Lyapunov function while optimizing the policy for stability.
"""
function update_neural_lac!(agent::NeuralLACAgent, obs, actions, rewards, next_obs)
    # 1. Update Lyapunov Manifold (Minimize Violation)
    gs_v = Zygote.gradient(agent.V_net) do v
        violation = LyapunovDrift(v, obs, next_obs; epsilon=0.01f0)
        # We also want V(0) = 0 and V(s) to be small
        return mean(violation.^2) + 1f-4 * mean(v(obs))
    end
    
    new_opt_v = agent.opt_state[3]
    new_v = agent.V_net
    if gs_v[1] !== nothing
        new_opt_v, new_v = Optimisers.update!(agent.opt_state[3], agent.V_net, gs_v[1])
    end

    # 2. Update Policy and Oracle
    gs_po = Zygote.gradient(agent.policy, agent.oracle) do p, o
        r_p = o(obs)
        μ, σ = p(obs)
        log_prob = -0.5f0 .* sum(((actions .- μ) ./ σ).^2 .+ 2.0f0 .* log.(σ))
        reward_loss = -mean(log_prob .* rewards)
        
        # Use LEARNED Lyapunov function for stability gap
        violation = LyapunovDrift(agent.V_net, obs, next_obs; epsilon=0.01f0)
        stability_loss = agent.λ_lyap * mean(violation)
        
        return reward_loss + stability_loss
    end
    
    new_opt_p = agent.opt_state[1]
    new_p = agent.policy
    if gs_po[1] !== nothing
        new_opt_p, new_p = Optimisers.update!(agent.opt_state[1], agent.policy, gs_po[1])
    end
    
    new_opt_o = agent.opt_state[2]
    new_o = agent.oracle
    if gs_po[2] !== nothing
        new_opt_o, new_o = Optimisers.update!(agent.opt_state[2], agent.oracle, gs_po[2])
    end
    
    agent.opt_state = (new_opt_p, new_opt_o, new_opt_v)
    agent.policy = new_p
    agent.oracle = new_o
    agent.V_net = new_v
end

end # module
