module FPRL

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export FPRLAgent, update_fprl!

mutable struct FPRLAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    opt_state
    lambda_fp::Float32
    FPRLAgent(o, p, ost, l) = new(o, p, ost, Float32(l))
end

# 1. New API
function FPRLAgent(state_dim::Int, out_dim::Int=1; lr=3e-4, lambda_fp=0.1f0)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    return FPRLAgent(o, p, lr, lambda_fp)
end

# 2. Legacy API
function FPRLAgent(o::ReflexiveOracle, p::GaussianPolicy, lr::Real=3e-4; lambda_fp=0.1f0)
    ost = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), (o.model, p.mu_net))
    return FPRLAgent(o, p, ost, lambda_fp)
end

function update_fprl!(agent::FPRLAgent, batch, env)
    # Target internal models for Zygote stability
    om = agent.oracle.model
    pm = agent.policy.mu_net
    
    gs = Zygote.gradient(om, pm) do o, p
        l_r = 0.0; l_f = 0.0
        for b in batch
            sc = b[1]; ac = b[2]; ret = b[6]
            rp = o(sc)
            mv = p(vcat(sc, rp))
            # Policy loss
            l_r -= sum((ac .- mv).^2) * ret
            # Fixed point proximity
            sp_pred = sc .+ mv .- Float32(env.alpha) .* rp
            l_f += sum((sc .- sp_pred).^2)
        end
        (l_r + agent.lambda_fp * l_f) / length(batch)
    end
    
    if !isnothing(gs) && gs[1] !== nothing && gs[2] !== nothing
        agent.opt_state, (agent.oracle.model, agent.policy.mu_net) = Optimisers.update!(agent.opt_state, (agent.oracle.model, agent.policy.mu_net), (gs[1], gs[2]))
    end
end

end # module
