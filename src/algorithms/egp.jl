module EGP

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export EGPAgent, update_egp!

mutable struct EGPAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    opt_state_oracle
    opt_state_policy
    beta_stab::Float32
end

function EGPAgent(state_dim::Int, out_dim::Int=1; lr_o=1e-4, lr_p=3e-4, beta_stab=0.1f0)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    oso = Optimisers.setup(Optimisers.Adam(lr_o), o)
    osp = Optimisers.setup(Optimisers.Adam(lr_p), p)
    EGPAgent(o, p, oso, osp, Float32(beta_stab))
end

function update_egp!(agent::EGPAgent, batch, env)
    gs = Zygote.gradient(agent.oracle, agent.policy) do orc, pol
        l_total = 0.0
        for b in batch
            s_val = b[1]
            rp_vec = orc(s_val); rp = rp_vec[1]
            mv, sv = pol(vcat(s_val, [rp]))
            # Saturated action for stability
            as = tanh.(mv) .+ sv .* 0.001f0
            sn = s_val .+ as .- Float32(env.alpha) .* rp
            l_total += -Interfaces.reward(env, sn, as) + agent.beta_stab * sum((sn .- s_val).^2)
        end
        return l_total / length(batch)
    end
    
    # Gradient clipping for high-end stability
    if gs[1] !== nothing
        g_clipped = Optimisers.clip(gs[1], 1.0f0)
        agent.opt_state_oracle, agent.oracle = Optimisers.update!(agent.opt_state_oracle, agent.oracle, g_clipped)
    end
    if gs[2] !== nothing
        g_clipped = Optimisers.clip(gs[2], 1.0f0)
        agent.opt_state_policy, agent.policy = Optimisers.update!(agent.opt_state_policy, agent.policy, g_clipped)
    end
end

end # module
