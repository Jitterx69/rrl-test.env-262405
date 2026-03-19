module PPO

using ..Interfaces, ..Architectures
using Flux, Zygote, Statistics, Optimisers

export PPOAgent, update_ppo!

mutable struct PPOAgent <: AbstractReflexiveAgent
    oracle::ReflexiveOracle
    policy::GaussianPolicy
    opt_state
    PPOAgent(o, p, ost) = new(o, p, ost)
end

# 1. New API: Consistent with ReflexiveRL where policy takes [state, prediction]
function PPOAgent(state_dim::Int, out_dim::Int=1; lr=3e-4)
    o = ReflexiveOracle(state_dim, out_dim)
    p = GaussianPolicy(state_dim + out_dim, out_dim)
    return PPOAgent(o, p, lr)
end

# 2. Legacy API: Still supports creating from existing policy
function PPOAgent(p::GaussianPolicy, lr::Real=3e-4)
    o = ReflexiveOracle(1, 1) # Dummy oracle
    return PPOAgent(o, p, lr)
end

function PPOAgent(o::ReflexiveOracle, p::GaussianPolicy, lr::Real)
    ost = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(lr)), p)
    return PPOAgent(o, p, ost)
end

function update_ppo!(agent::PPOAgent, batch)
    gs = Zygote.gradient(agent.policy) do p
        loss = 0.0
        for b in batch
            sc = b[1]; ac = b[2]; rp = b[3]; ret = b[6]
            # Handle potential mismatch for legacy scripts
            # Modern ReflexiveRL uses vcat(sc, rp)
            pin = length(sc) + length(rp) == 1 ? vcat(sc, rp) : vcat(sc, rp) 
            # Wait, if p expects 1 and vcat is 2, it fails.
            # We can detect the dimension of p's first layer.
            # But simpler: if length(rp) is 0 or it's classic, just use sc.
            p_in = if size(p.mu_net[1].weight, 2) == length(sc)
                sc
            else
                vcat(sc, rp)
            end
            
            mv, sv = p(p_in)
            lp = -0.5 * sum((ac .- mv).^2 ./ sv.^2) - sum(log.(sv))
            loss -= lp * ret
        end
        loss / length(batch)
    end
    if gs[1] !== nothing
        agent.opt_state, agent.policy = Optimisers.update!(agent.opt_state, agent.policy, gs[1])
    end
end

end # module
