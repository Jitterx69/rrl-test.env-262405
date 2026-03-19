module ICRL

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export ICRLAgent, update_icrl!

mutable struct ICRLAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    opt_state
    beta::Float32
    ICRLAgent(o, p, ost, b) = new(o, p, ost, Float32(b))
end

# 1. New API
function ICRLAgent(state_dim::Int, out_dim::Int=1; lr=1e-4, beta=0.01f0)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    return ICRLAgent(o, p, lr, beta)
end

# 2. Legacy API
function ICRLAgent(o::ReflexiveOracle, p::GaussianPolicy, lr::Real=1e-4; beta=0.01f0)
    ost = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), (o, p))
    return ICRLAgent(o, p, ost, beta)
end

function update_icrl!(agent::ICRLAgent, batch)
    gs = Zygote.gradient(agent.oracle, agent.policy) do o, p
        loss = 0.0
        for b in batch
            sc = b[1]; ac = b[2]; ret = b[6]
            rp = o(sc)
            mv, sv = p(vcat(sc, rp))
            lp = -0.5 * sum((ac .- mv).^2 ./ sv.^2) - sum(log.(sv))
            loss -= lp * ret 
            loss += agent.beta * sum(rp.^2)
        end
        loss / length(batch)
    end
    if gs[1] !== nothing && gs[2] !== nothing
        agent.opt_state, (agent.oracle, agent.policy) = Optimisers.update!(agent.opt_state, (agent.oracle, agent.policy), (gs[1], gs[2]))
    end
end

end # module
