# Monolithic Verification Script (Fixed)
using Pkg; Pkg.activate(".")
using Flux, Zygote, Statistics, Printf, Random

# 1. Models
struct Oracle; m; end; Flux.@layer Oracle
function create_oracle(in_dim, out_dim=1)
    Oracle(Chain(Dense(in_dim, 64, relu), Dense(64, out_dim)))
end
(o::Oracle)(x) = o.m(x)

struct Policy; m; ls; end; Flux.@layer Policy
function create_policy(in_dim, out_dim=1)
    Policy(Chain(Dense(in_dim, 64, relu), Dense(64, out_dim)), fill(-2.0f0, out_dim))
end
(p::Policy)(x) = (p.m(x), exp.(p.ls))

# 2. Env
mutable struct Env; alpha; state; end
reset!(e) = (e.state = 0.0f0; e.state)
function step_env!(e, a, rp)
    noise = 0.05f0 * randn(Float32)
    e.state = e.state + Float32(a) - Float32(e.alpha)*Float32(rp) + noise
    return e.state
end
reward_fn(s, a) = -(s^2) - 0.1f0*a^2

# 3. Main Loop
function run_monolithic()
    o = create_oracle(1)
    p = create_policy(2)
    ev = Env(1.0, 0.0f0)
    opt_o = Flux.setup(Adam(1e-4), o)
    opt_p = Flux.setup(Adam(3e-4), p)
    
    println(">>> Starting Monolithic Training (100 epochs)...")
    for ep in 1:100
        batch = []
        s = reset!(ev)
        for t in 1:100
            rp = o([s])[1]
            mv, sv = p([s, rp])
            a = mv[1] + sv[1]*randn(Float32)
            sn = step_env!(ev, a, rp)
            r = reward_fn(sn, a)
            push!(batch, (s, a, rp, sn, r))
            s = sn
        end
        
        # Returns
        G = reverse(cumsum(reverse([b[5] for b in batch])))
        
        gs = Zygote.gradient(o, p) do orc, pol
            l = 0.0
            for i in 1:length(batch)
                s_v, a_true, rp_v, sn_v, r_v = batch[i]
                rp_sim = orc([s_v])[1]
                mv_sim, sv_sim = pol([s_v, rp_sim])
                a_sim = mv_sim[1] + sv_sim[1]*0.001f0
                sn_sim = s_v + a_sim - 1.0f0 * rp_sim
                l += -reward_fn(sn_sim, a_sim)
            end
            l / length(batch)
        end
        Flux.update!(opt_o, o, gs[1])
        Flux.update!(opt_p, p, gs[2])
    end
    
    # Eval
    s = reset!(ev); tr = 0.0; se = 0.0
    for _ in 1:1000
        rp = o([s])[1]; mv, _ = p([s, rp]); a = mv[1]
        sn = step_env!(ev, a, rp)
        tr += reward_fn(sn, a)
        se += abs(sn - (s + a - ev.alpha*rp))
        s = sn
    end
    @printf("\n=== Results ===\n")
    @printf("Mean Reward: %.3f\n", tr/1000)
    @printf("Stab Error: %.4f\n", se/1000)
end

run_monolithic()
