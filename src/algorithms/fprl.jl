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
    ost = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), (o, p))
    return FPRLAgent(o, p, ost, lambda_fp)
end

function update_fprl!(agent::FPRLAgent, batch, env)
    gs = Zygote.gradient(agent.oracle, agent.policy) do o, p
        l_r = 0.0; l_f = 0.0
        for b in batch
            sc = b[1]; ac = b[2]; ret = b[6]
            rp = o(sc)
            mv, sv = p(vcat(sc, rp))
            lp = -0.5 * sum((ac .- mv).^2 ./ sv.^2) - sum(log.(sv))
            l_r -= lp * ret
            # Fixed point proximity: || s - (s + a - alpha * rho) || = || a - alpha * rho ||
            # Or simplified here to || s - s_next_pred ||
            sp_pred = sc .+ mv .- Float32(env.alpha) .* rp
            l_f += sum((sc .- sp_pred).^2)
        end
        (l_r + agent.lambda_fp * l_f) / length(batch)
    end
    if gs[1] !== nothing && gs[2] !== nothing
        agent.opt_state, (agent.oracle, agent.policy) = Optimisers.update!(agent.opt_state, (agent.oracle, agent.policy), (gs[1], gs[2]))
    end
end

end # module
