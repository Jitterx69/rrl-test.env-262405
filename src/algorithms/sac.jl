module SAC

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export SACAgent, update_sac!

mutable struct SACAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    critic1
    critic2
    opt_state_policy
    opt_state_critic
    log_alpha::Vector{Float32}
    target_entropy::Float32
    # Internal constructor to prevent precompilation overwriting
    SACAgent(o, p, c1, c2, osp, osc, la, te) = new(o, p, c1, c2, osp, osc, la, Float32(te))
end

# 1. New API (Canonical)
function SACAgent(state_dim::Int, out_dim::Int=1; lr=3e-4, initial_log_alpha=0.0f0)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    
    # Dual Critics
    cin = state_dim + out_dim + out_dim # state + rp + action
    c1 = Chain(Dense(cin, 64, relu), Dense(64, 64, relu), Dense(64, 1))
    c2 = Chain(Dense(cin, 64, relu), Dense(64, 64, relu), Dense(64, 1))
    
    osp = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), p)
    osc = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), (c1, c2))
    
    la = [Float32(initial_log_alpha)]
    te = -Float32(out_dim)
    
    return SACAgent(o, p, c1, c2, osp, osc, la, te)
end

function SACAgent(o::ReflexiveOracle, p::GaussianPolicy, c1, c2, lr::Real=3e-4; initial_log_alpha=0.0f0)
    osp = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), p)
    osc = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), (c1, c2))
    la = [Float32(initial_log_alpha)]
    te = -Float32(size(p.mu_net[end].weight, 1))
    return SACAgent(o, p, c1, c2, osp, osc, la, te)
end

function (a::SACAgent)(obs)
    # 1. Oracle prediction
    r_p = a.oracle(obs)
    # 2. Policy action (mu + sigma * noise)
    μ, σ = a.policy(vcat(collect(obs), collect(r_p)))
    action = μ .+ σ .* randn(Float32, size(μ))
    return action, r_p
end

function update_sac!(agent::SACAgent, batch)
    # Correct handling of [state, prediction, action, next_state, reward, return]
    # SAC typically uses current reward + next_state value, or direct returns for simplicity in this baseline.
    
    # 1. Critic Update
    gs_c = Zygote.gradient(agent.critic1, agent.critic2) do c1, c2
        l = 0.0
        for b in batch
            sc=b[1]; act=b[2]; rp=b[3]; ret=b[6]
            # Architecture expects [state, prediction, action]
            cin = vcat(sc, rp, act)
            l += (c1(cin)[1] - ret)^2 + (c2(cin)[1] - ret)^2
        end
        l / length(batch)
    end
    if gs_c[1] !== nothing && gs_c[2] !== nothing
        agent.opt_state_critic, (agent.critic1, agent.critic2) = Optimisers.update!(agent.opt_state_critic, (agent.critic1, agent.critic2), (gs_c[1], gs_c[2]))
    end
    
    # 2. Policy Update
    gs_p = Zygote.gradient(agent.policy) do pol
        l = 0.0
        alpha = exp(agent.log_alpha[1])
        for b in batch
            sc=b[1]; rp=b[3]
            mv, sv = pol(vcat(sc, rp))
            # Reparameterization trick action
            act_samp = mv .+ sv .* randn(Float32, length(mv))
            # Current value (min-Q)
            q_val = min(agent.critic1(vcat(sc, rp, act_samp))[1], 
                        agent.critic2(vcat(sc, rp, act_samp))[1])
            # SAC loss: - (Q - alpha * log_prob)
            # Simplified log_prob for Gaussian: 
            l_prob = -0.5 * sum((act_samp .- mv).^2 ./ sv.^2) - sum(log.(sv))
            l -= (q_val - alpha * l_prob)
        end
        l / length(batch)
    end
    if gs_p[1] !== nothing
        agent.opt_state_policy, agent.policy = Optimisers.update!(agent.opt_state_policy, agent.policy, gs_p[1])
    end
end

end # module
