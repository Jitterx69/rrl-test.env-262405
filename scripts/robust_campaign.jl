# Robust High-End Campaign Script
using Pkg; Pkg.activate(".")
using Flux, Zygote, Statistics, Printf, Random

# 1. Component Loading
include("../src/core/interfaces.jl")
include("../src/models/architectures.jl")

# 2. Agent Logic (Self-Contained for Verification)
struct CampaignAgent
    o; p; oso; osp
end

function create_agent(sd=1, od=1)
    o = Architectures.ReflexiveOracle(sd, od)
    p = Architectures.GaussianPolicy(sd + od, od)
    CampaignAgent(o, p, Flux.setup(Adam(1e-4), o), Flux.setup(Adam(3e-4), p))
end

function update!(a::CampaignAgent, batch)
    gs = Zygote.gradient(a.o.model, a.p.mu_net, a.p.log_sigma) do om, pm, ls
        l = 0.0
        for b in batch
            s=b[1]; a_t=b[2]; rp=b[3]; ret=b[6]
            # Sim
            r_sim = om(s)[1]
            m_s = pm(vcat(s, [r_sim]))
            mv = m_s[1]; sv = exp.(ls)[1]
            a_sim = mv + sv * 0.001f0
            sn_sim = s[1] + a_sim - 1.0f0 * r_sim
            l += (sn_sim^2) + 0.1f0*a_sim^2 # Minimize state and energy
        end
        l / length(batch)
    end
    Flux.update!(a.oso, a.o.model, gs[1])
    Flux.update!(a.osp, a.p.mu_net, gs[2])
    a.p.log_sigma .-= 1e-4 .* gs[3] # Direct update for vector
end

# 3. Execution
function run_stats(n=10)
    println(">>> Starting Robust Campaign ($n runs)...")
    rewards = []
    for i in 1:n
        agent = create_agent()
        for ep in 1:50
            batch = []
            s = [0.0f0]
            for _ in 1:100
                rp = agent.o(s)[1]
                mv, sv = agent.p(vcat(s, [rp]))
                act = mv[1] + sv[1]*randn(Float32)
                sn = [s[1] + act - 1.0f0*rp + 0.05f0*randn(Float32)]
                r = -(sn[1]^2) - 0.1f0*act^2
                push!(batch, (s, [act], [rp], sn, r, 0.0f0))
                s = sn
            end
            rets = reverse(cumsum(reverse([b[5] for b in batch])))
            update!(agent, [(b[1], b[2], b[3], b[4], b[5], rets[j]) for (j, b) in enumerate(batch)])
        end
        # Eval
        s = [0.0f0]; er = 0.0
        for _ in 1:200
            rp = agent.o(s)[1]; mv, _ = agent.p(vcat(s, [rp])); a=mv[1]
            sn = [s[1] + a - 1.0f0*rp + 0.05f0*randn(Float32)]
            er += -(sn[1]^2) - 0.1f0*a^2
            s = sn
        end
        push!(rewards, er/200)
        print(".")
    end
    @printf("\nFinal Result: Avg R = %.3f | Std = %.3f\n", mean(rewards), std(rewards))
end

run_stats(10)
