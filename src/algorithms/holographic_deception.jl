module HolographicDeception

using Flux, Zygote, Optimisers, Statistics, LinearAlgebra
using ..Architectures, ..Adversarial, ..Interfaces

export HolographicDeceiver, update_holographic!

"""
    HolographicDeceiver(s_dim, a_dim, hidden_dim, modes)

A state-of-the-art adversarial agent that uses Gated Fourier Units 
to generate "spectral decoys"—deceptive signals that mirror natural system frequencies.
"""
mutable struct HolographicDeceiver <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    spectral_poisoner::GatedSpectralOracle # Uses FNO+ for deception
    opt_state_o
    opt_state_p
    opt_state_sp
    beta_stab::Float32
    lambda_adv::Float32
end

function HolographicDeceiver(s_dim::Int, a_dim::Int; hidden_dim=64, modes=16, lr=3e-4, beta=0.1f0, lambda=0.2f0)
    o = ReflexiveOracle(s_dim, a_dim)
    p = GaussianPolicy(s_dim + a_dim, a_dim)
    # Spectral Poisoner captures global patterns to generate "natural-looking" noise
    sp = GatedSpectralOracle(s_dim + a_dim, a_dim, hidden_dim, modes)
    
    oso = Optimisers.setup(Optimisers.Adam(lr), o.model)
    osp = Optimisers.setup(Optimisers.Adam(lr), p.mu_net)
    ossp = Optimisers.setup(Optimisers.Adam(lr), sp.model)
    
    return HolographicDeceiver(o, p, sp, oso, osp, ossp, Float32(beta), Float32(lambda))
end

function (agent::HolographicDeceiver)(s; adversarial=true)
    s_v = s isa AbstractVector ? Float32.(s) : [Float32(s)]
    rp = agent.oracle(s_v)
    
    if adversarial
        # Spectral deception: generate noise that is frequency-aligned with the state
        poison = agent.spectral_poisoner(vcat(s_v, vec(rp)))
        rp = rp .+ poison
    end
    
    input_p = vcat(s_v, vec(rp))
    mv, sv = agent.policy(input_p)
    return mv, sv
end

function update_holographic!(agent::HolographicDeceiver, batch, env)
    om = agent.oracle.model
    pm = agent.policy.mu_net
    spm = agent.spectral_poisoner.model
    
    gs = Zygote.gradient(om, pm, spm) do o, p, sp
        total_loss = 0.0f0
        for b in batch
            s_c = b[1]; a_c = b[2]; r_c = b[6]
            rp = o(s_c)
            poison = sp(vcat(s_c, vec(rp)))
            rp_adv = rp .+ poison
            
            mv = p(vcat(s_c, vec(rp_adv)))
            l_p = -sum((a_c .- mv).^2) * r_c
            
            # Stability constraint
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
            agent.opt_state_sp, agent.spectral_poisoner.model = Optimisers.update!(agent.opt_state_sp, agent.spectral_poisoner.model, gs[3])
        end
    end
end

end # module
