module Adversarial

using Flux, Zygote, Optimisers, Statistics, LinearAlgebra
using ..Architectures, ..Interfaces

export AdversarialReflexiveAgent, update_adversarial!

"""
    AdversarialReflexiveAgent(oracle, policy, poisoner)

An advanced agent that can manipulate its own reflexive signals to deceive competitors.
The `poisoner` is a neural network that generates a bias to add to the oracle signal.
"""
mutable struct AdversarialReflexiveAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    poisoner::Chain # Generates adversarial signal perturbation
    opt_state_o
    opt_state_p
    opt_state_pos
    beta_stab::Float32 # Stability weight
    lambda_adv::Float32 # Adversarial deception weight
end

function AdversarialReflexiveAgent(s_dim::Int, a_dim::Int; lr=3e-4, beta=0.1f0, lambda=0.1f0)
    o = ReflexiveOracle(s_dim, a_dim)
    p = GaussianPolicy(s_dim + a_dim, a_dim)
    # Poisoner takes state and oracle output, returns a perturbation of the same dim as oracle
    pos = Chain(Dense(s_dim + a_dim, 32, relu), Dense(32, a_dim))
    
    oso = Optimisers.setup(Optimisers.Adam(lr), o.model)
    osp = Optimisers.setup(Optimisers.Adam(lr), p.mu_net)
    oppos = Optimisers.setup(Optimisers.Adam(lr), pos)
    
    return AdversarialReflexiveAgent(o, p, pos, oso, osp, oppos, Float32(beta), Float32(lambda))
end

function (agent::AdversarialReflexiveAgent)(s; adversarial=true)
    s_v = s isa AbstractVector ? Float32.(s) : [Float32(s)]
    rp = agent.oracle(s_v)
    
    if adversarial
        # Add deception signal: rho_adv = rho + poisoner(s, rho)
        poison = agent.poisoner(vcat(s_v, vec(rp)))
        rp = rp .+ poison
    end
    
    input_p = vcat(s_v, vec(rp))
    mv, sv = agent.policy(input_p)
    return mv, sv
end

function update_adversarial!(agent::AdversarialReflexiveAgent, batch, env)
    om = agent.oracle.model
    pm = agent.policy.mu_net
    posm = agent.poisoner
    
    # Combined gradient for adversarial objective
    # Objective: Maximize reward + beta*Stability - lambda*CompetitorUtility
    # In this phase, we simplify to: reward + beta*Stability while making the signal harder to predict
    gs = Zygote.gradient(om, pm, posm) do o, p, pos
        total_loss = 0.0f0
        for b in batch
            s_c = b[1]; a_c = b[2]; r_c = b[6]
            # Standard reflexive loss
            rp = o(s_c)
            poison = pos(vcat(s_c, vec(rp)))
            rp_adv = rp .+ poison
            
            mv = p(vcat(s_c, vec(rp_adv)))
            # Policy improvement
            l_p = -sum((a_c .- mv).^2) * r_c
            
            # Deception loss: we want rp_adv to be different from the "natural" fixed point
            # to make it harder for competitors to use as a proxy for our state
            # but also keep the system from diverging
            sp_pred = s_c .+ mv .- Float32(env.alpha) .* rp_adv
            l_stab = sum((s_c .- sp_pred).^2)
            
            total_loss += l_p + agent.beta_stab * l_stab
        end
        total_loss / length(batch)
    end
    
    if !isnothing(gs)
        if !isnothing(gs[1])
            agent.opt_state_o, agent.oracle.model = Optimisers.update!(agent.opt_state_o, agent.oracle.model, gs[1])
        end
        if !isnothing(gs[2])
            agent.opt_state_p, agent.policy.mu_net = Optimisers.update!(agent.opt_state_p, agent.policy.mu_net, gs[2])
        end
        if !isnothing(gs[3])
            agent.opt_state_pos, agent.poisoner = Optimisers.update!(agent.opt_state_pos, agent.poisoner, gs[3])
        end
    end
end

end # module
