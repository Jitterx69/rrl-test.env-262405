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
    # Internal constructor to avoid default overwriting
    EGPAgent(o, p, oso, osp, b) = new(o, p, oso, osp, Float32(b))
end

# 1. New API (Recommended)
function EGPAgent(state_dim::Int, out_dim::Int=1; lr_o=1e-4, lr_p=3e-4, beta_stab=0.1f0)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    return EGPAgent(o, p, lr_o, lr_p, beta_stab)
end

# 2. Legacy/Flexible API
# Disambiguate by using Keyword for beta_stab and ensuring 3 args don't conflict with 5 args
function EGPAgent(o::ReflexiveOracle, p::GaussianPolicy, lr::Real; beta_stab=0.1f0)
    return EGPAgent(o, p, lr, lr, beta_stab)
end

function EGPAgent(o::ReflexiveOracle, p::GaussianPolicy, lr_o::Real, lr_p::Real, b_stab::Real=0.1f0)
    oso = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr_o)), o)
    osp = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr_p)), p)
    return EGPAgent(o, p, oso, osp, b_stab)
end

function update_egp!(agent::EGPAgent, batch, env)
    gs = Zygote.gradient(agent.oracle, agent.policy) do orc, pol
        l_total = 0.0
        for b in batch
            s_val = b[1]
            rp_vec = orc(s_val); rp = rp_vec[1]
            mv, sv = pol(vcat(s_val, [rp]))
            as = tanh.(mv) .+ sv .* 0.001f0
            sn = s_val .+ as .- Float32(env.alpha) .* rp
            l_total += -Interfaces.reward(env, sn, as) + agent.beta_stab * sum((sn .- s_val).^2)
        end
        return l_total / length(batch)
    end
    
    if gs[1] !== nothing
        agent.opt_state_oracle, agent.oracle = Optimisers.update!(agent.opt_state_oracle, agent.oracle, gs[1])
    end
    if gs[2] !== nothing
        agent.opt_state_policy, agent.policy = Optimisers.update!(agent.opt_state_policy, agent.policy, gs[2])
    end
end

end # module
