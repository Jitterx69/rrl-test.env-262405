module PPO

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export PPOAgent, update_ppo!

mutable struct PPOAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    opt_state
end

function PPOAgent(state_dim::Int, out_dim::Int=1; lr=3e-4)
    o = ReflexiveOracle(state_dim, out_dim) # Only used for baseline compatibility if needed
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    ost = Optimisers.setup(Optimisers.Adam(lr), p)
    PPOAgent(o, p, ost)
end

function update_ppo!(agent::PPOAgent, batch)
    gs = Zygote.gradient(agent.policy) do p
        loss = 0.0
        for b in batch
            sc = b[1]; ac = b[2]; rp = b[3]; ret = b[6]
            mv, sv = p(vcat(sc, rp))
            lp = -0.5 * sum((ac .- mv).^2 ./ sv.^2) - sum(log.(sv))
            loss -= lp * ret
        end
        loss / length(batch)
    end
    if gs[1] !== nothing
        gc = Optimisers.clip(gs[1], 1.0f0)
        agent.opt_state, agent.policy = Optimisers.update!(agent.opt_state, agent.policy, gc)
    end
end

end # module
