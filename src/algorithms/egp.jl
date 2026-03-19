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
    EGPAgent(o, p, oso, osp, b) = new(o, p, oso, osp, Float32(b))
end

# 1. New API
function EGPAgent(state_dim::Int, out_dim::Int=1; lr_o=1e-4, lr_p=3e-4, beta_stab=0.1f0)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    EGPAgent(o, p, lr_o, lr_p, beta_stab)
end

# 2. Legacy API
function EGPAgent(o::ReflexiveOracle, p::GaussianPolicy, lr::Real; beta_stab=0.1f0)
    EGPAgent(o, p, lr, lr, beta_stab)
end

function EGPAgent(o::ReflexiveOracle, p::GaussianPolicy, lr_o::Real, lr_p::Real, b_stab::Real=0.1f0)
    oso = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr_o)), o.model)
    osp = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr_p)), p.mu_net)
    EGPAgent(o, p, oso, osp, b_stab)
end

function update_egp!(agent::EGPAgent, batch, env)
    # 1. Update Oracle Internal Model
    orc_m = agent.oracle.model
    gs_o = Zygote.gradient(orc_m) do m
        l_total = 0.0
        for b in batch
            s_val = b[1]
            rp_vec = m(s_val)
            sn = s_val .+ b[2] .- Float32(env.alpha) .* rp_vec
            l_total += agent.beta_stab * sum((sn .- s_val).^2)
        end
        return l_total / length(batch)
    end
    
    # 2. Update Policy Internal Models
    pol_m = agent.policy.mu_net
    gs_p = Zygote.gradient(pol_m) do m
        l_total = 0.0
        for b in batch
            s_val = b[1]
            rp_vec = agent.oracle(s_val)
            mv = m(vcat(s_val, rp_vec))
            as = mv 
            sn = s_val .+ as .- Float32(env.alpha) .* rp_vec
            l_total += -Interfaces.reward(env, sn, as)
        end
        return l_total / length(batch)
    end
    
    if !isnothing(gs_o) && !isnothing(gs_o[1])
        agent.opt_state_oracle, agent.oracle.model = Optimisers.update!(agent.opt_state_oracle, agent.oracle.model, gs_o[1])
    end
    if !isnothing(gs_p) && !isnothing(gs_p[1])
        agent.opt_state_policy, agent.policy.mu_net = Optimisers.update!(agent.opt_state_policy, agent.policy.mu_net, gs_p[1])
    end
end

end # module
