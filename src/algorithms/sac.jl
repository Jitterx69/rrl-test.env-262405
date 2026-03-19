module SAC

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export SACAgent, update_sac!

mutable struct SACAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy # Renamed from actor for consistency
    critic1
    critic2
    opt_state_policy # Renamed from opt_state_actor
    opt_state_critic
    log_alpha
    target_entropy
end

function SACAgent(state_dim::Int, out_dim::Int=1; lr=1e-5)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    cin = state_dim + out_dim + out_dim
    c1 = Chain(Dense(cin, 64, relu), Dense(64, 64, relu), Dense(64, 1))
    c2 = Chain(Dense(cin, 64, relu), Dense(64, 64, relu), Dense(64, 1))
    la = [0.0f0]
    te = -Float32(out_dim)
    osp = Optimisers.setup(Optimisers.Adam(lr), p)
    osc = Optimisers.setup(Optimisers.Adam(lr), (c1, c2))
    SACAgent(o, p, c1, c2, osp, osc, la, te)
end

function update_sac!(agent::SACAgent, batch)
    # Critic Update
    gs_c = Zygote.gradient(agent.critic1, agent.critic2) do c1, c2
        l = 0.0
        for b in batch
            sc=b[1]; act=b[2]; rp=b[3]; ret=b[6]
            l += (c1(vcat(sc, rp, act))[1] - ret)^2 + (c2(vcat(sc, rp, act))[1] - ret)^2
        end
        l / length(batch)
    end
    gc1 = Optimisers.clip(gs_c[1], 1.0f0); gc2 = Optimisers.clip(gs_c[2], 1.0f0)
    agent.opt_state_critic, (agent.critic1, agent.critic2) = Optimisers.update!(agent.opt_state_critic, (agent.critic1, agent.critic2), (gc1, gc2))
    
    # Policy Update (Renamed from Actor)
    gs_p = Zygote.gradient(agent.policy) do pol_net
        l = 0.0
        for b in batch
            sc=b[1]; rp=b[3]
            mv, sv = pol_net(vcat(sc, rp))
            alpha = exp(agent.log_alpha[1])
            l -= mv[1] * agent.critic1(vcat(sc, rp, mv))[1] + alpha * sum(log.(sv))
        end
        l / length(batch)
    end
    gp = Optimisers.clip(gs_p[1], 1.0f0)
    agent.opt_state_policy, agent.policy = Optimisers.update!(agent.opt_state_policy, agent.policy, gp)
end

end # module
