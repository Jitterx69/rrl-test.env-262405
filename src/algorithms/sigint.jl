module SIGINT

using Flux, Zygote, Optimisers, Statistics, LinearAlgebra
using ..Architectures, ..Interfaces

export SIGINTAgent, update_sigint!

"""
    SIGINTAgent(oracle, policy, filter)

A sophisticated agent that can intercept and decode poisoned reflexive signals.
Uses a spectral de-noising filter to recover the "clean" state-prediction vector.
"""
mutable struct SIGINTAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle # Local internal oracle
    policy::GaussianPolicy
    denoiser::SpectralOracle # Filters intercepted signal
    opt_state_o
    opt_state_p
    opt_state_d
end

function SIGINTAgent(s_dim::Int, a_dim::Int; hidden_dim=64, modes=16, lr=3e-4)
    o = ReflexiveOracle(s_dim, a_dim)
    p = GaussianPolicy(s_dim + a_dim, a_dim)
    # Denoiser takes intercepted (noisy) signal and state, returns clean signal
    d = SpectralOracle(s_dim + a_dim, a_dim, hidden_dim, modes)
    
    oso = Optimisers.setup(Optimisers.Adam(lr), o.model)
    osp = Optimisers.setup(Optimisers.Adam(lr), p.mu_net)
    osd = Optimisers.setup(Optimisers.Adam(lr), d.model)
    
    return SIGINTAgent(o, p, d, oso, osp, osd)
end

function (agent::SIGINTAgent)(s, intercepted_signal)
    s_v = s isa AbstractArray ? Float32.(s) : [Float32(s)]
    is_v = intercepted_signal isa AbstractArray ? Float32.(intercepted_signal) : [Float32(intercepted_signal)]
    
    # De-noise the intercepted signal to reveal true intent
    clean_rp = agent.denoiser(vcat(s_v, vec(is_v)))
    
    input_p = vcat(s_v, vec(clean_rp))
    mv, sv = agent.policy(input_p)
    return mv, sv
end

function update_sigint!(agent::SIGINTAgent, batch, target_clean_signals)
    om = agent.oracle.model
    pm = agent.policy.mu_net
    dm = agent.denoiser.model
    
    # De-noising objective: Match recovered signal to actual clean signals (if supervised)
    # or maximize self-consistency.
    gs = Zygote.gradient(om, pm, dm) do o, p, d
        total_loss = 0.0f0
        for (i, b) in enumerate(batch)
            s_c = b[1]; a_c = b[2]; r_c = b[6]
            is_c = b[3] # Intercepted (poisoned) signal
            
            recovered = d(vcat(s_c, vec(is_c)))
            # Supervised de-noising if target is known
            l_d = sum((recovered .- target_clean_signals[i]).^2)
            
            mv = p(vcat(s_c, vec(recovered)))
            l_p = -sum((a_c .- mv).^2) * r_c
            
            total_loss += l_p + 1.0f0 * l_d
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
            agent.opt_state_d, agent.denoiser.model = Optimisers.update!(agent.opt_state_d, agent.denoiser.model, gs[3])
        end
    end
end

end # module
