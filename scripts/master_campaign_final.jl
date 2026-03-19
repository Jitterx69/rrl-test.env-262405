# Final High-End Master Campaign Script
# Performance Optimized | Absolute AD Stability | Multi-Tier Reporting

using Pkg; Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src"))

using ReflexiveRL
using Flux, Zygote, Statistics, Random, CSV, DataFrames, Printf

# 1. Verification Logic
function run_single_run(algo, tier, seed=1337)
    Random.seed!(seed)
    
    # Env & Agent setup
    env = if tier == 1; Tier1Env(1.0, 0.05)
    elseif tier == 2; Tier2Env(5, 0.5)
    else; Tier3Env(0.1, 0.5) end
    
    sd = (tier == 3 ? 2 : 1)
    agent = if algo == "EGP"; EGPAgent(sd, 1)
    elseif algo == "FPRL"; FPRLAgent(sd, 1)
    elseif algo == "PPO"; PPOAgent(sd, 1)
    else; return nothing end

    # Training
    for ep in 1:40
        batch = []
        s = reset!(env)
        for t in 1:100
            sv = (s isa Vector ? s : [Float32(s)])
            rp = agent.oracle(sv)[1]
            mu, sigma = agent.policy(vcat(sv, [rp]))
            a = mu[1] + sigma[1]*randn(Float32)
            sn = step!(env, a, rp)
            push!(batch, (sv, [a], [rp], (sn isa Vector ? sn : [sn]), reward(env, sn, a), 0.0f0))
            s = sn
        end
        # Updates are now stable with explicit parameter gradients
        if algo == "EGP"; update_egp!(agent, batch, env)
        elseif algo == "FPRL"; update_fprl!(agent, batch, env)
        elseif algo == "PPO"; update_ppo!(agent, batch) end
    end
    
    # Eval
    er_sum = 0.0
    s = reset!(env)
    for _ in 1:100
        sv = (s isa Vector ? s : [Float32(s)])
        rp = agent.oracle(sv)[1]
        mu, _ = agent.policy(vcat(sv, [rp]))
        a = mu[1]
        sn = step!(env, a, rp)
        er_sum += reward(env, sn, a)
        s = sn
    end
    return er_sum / 100
end

# 2. Master Campaign
function run_master_campaign(n=20)
    println(">>> Starting Master High-End Campaign ($n runs per config)")
    results = []
    
    for tier in [1, 2, 3]
        for algo in ["EGP", "FPRL", "PPO"]
            print("Tier $tier - $algo: ")
            rs = [run_single_run(algo, tier, 2000 + i) for i in 1:n]
            m = mean(rs); s = std(rs)
            @printf(" Avg R: %.3f\n", m)
            push!(results, (Tier=tier, Algo=algo, Reward=m, Std=s))
        end
    end
    
    CSV.write("experiments/results/final_verification.csv", DataFrame(results))
    println(">>> Report saved to experiments/results/final_verification.csv")
end

run_master_campaign(5) # Small n for final check
