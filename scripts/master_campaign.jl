# Definitive Verification Campaign (High-End ReflexiveRL)
# 20 Runs per configuration (Total 300 runs) - Sequential for absolute stability

using Pkg
Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src"))

using ReflexiveRL
using Statistics, Random, CSV, DataFrames, Dates, Printf

# Explicitly import everything to avoid namespace issues
import ReflexiveRL: EGPAgent, FPRLAgent, PPOAgent, ICRLAgent, SACAgent
import ReflexiveRL: ReflexiveTrainer, reset!, step!, reward, compute_returns
import ReflexiveRL: update_egp!, update_fprl!, update_ppo!, update_icrl!, update_sac!
import ReflexiveRL: Tier1Env, Tier2Env, Tier3Env

function run_experimental_unit(algo_name, tier_idx, seed)
    Random.seed!(seed)
    
    # 1. Setup Env
    env = if tier_idx == 1; Tier1Env(1.0, 0.05)
    elseif tier_idx == 2; Tier2Env(5, 0.5)
    else; Tier3Env(0.1, 0.5) end
    
    st_dim = tier_idx == 3 ? 2 : 1
    
    # 2. Setup Agent
    agent = if algo_name == "EGP"; EGPAgent(st_dim, 1)
    elseif algo_name == "FPRL"; FPRLAgent(st_dim, 1)
    elseif algo_name == "PPO"; PPOAgent(st_dim, 1)
    elseif algo_name == "ICRL"; ICRLAgent(st_dim, 1)
    elseif algo_name == "SAC"; SACAgent(st_dim, 1)
    else; return nothing end
    
    # 3. Train
    epochs = 40
    steps = 100
    for ep in 1:epochs
        batch = []
        s = reset!(env)
        for t in 1:steps
            sv = s isa Vector ? s : [Float32(s)]
            rp = agent.oracle(sv)[1]
            pin = vcat(sv, [Float32(rp)])
            mv, sv_p = agent.policy(pin)
            a = mv[1] + sv_p[1] * randn(Float32)
            sn = step!(env, a, rp)
            r = reward(env, sn, a)
            snv = sn isa Vector ? sn : [Float32(sn)]
            push!(batch, (sv, [Float32(a)], [Float32(rp)], snv, Float32(r), 0.0f0))
            s = sn
        end
        
        rtns = compute_returns([b[5] for b in batch])
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(rtns[i]))
        end
        
        if algo_name == "EGP"; update_egp!(agent, batch, env)
        elseif algo_name == "FPRL"; update_fprl!(agent, batch, env)
        elseif algo_name == "PPO"; update_ppo!(agent, batch)
        elseif algo_name == "ICRL"; update_icrl!(agent, batch)
        elseif algo_name == "SAC"; update_sac!(agent, batch) end
    end
    
    # 4. Eval
    er_sum = 0.0
    s = reset!(env)
    for _ in 1:100
        sv = s isa Vector ? s : [Float32(s)]
        rp = agent.oracle(sv)[1]
        pin = vcat(sv, [Float32(rp)])
        mv, _ = agent.policy(pin)
        a = mv[1]
        sn = step!(env, a, rp)
        er_sum += reward(env, sn, a)
        s = sn
    end
    return er_sum / 100
end

function master_campaign(n=20)
    algos = ["EGP", "FPRL", "PPO", "ICRL", "SAC"]
    tiers = [1, 2, 3]
    results = []
    
    println(">>> Starting Master Campaign ($n runs per config - Sequential)")
    
    for tier in tiers
        for algo in algos
            print("Tier $tier - $algo: ")
            rs = zeros(n)
            for i in 1:n
                try
                    rs[i] = run_experimental_unit(algo, tier, 2000 + i)
                    if i % 5 == 0; print("."); end
                catch e
                    println("\n[ERROR] Tier $tier - $algo - Run $i failed: $e")
                    rethrow(e)
                end
            end
            m = mean(rs); s = std(rs)
            @printf(" Avg R: %.3f | Std: %.3f\n", m, s)
            push!(results, (Tier=tier, Algo=algo, Reward=m, Std=s))
        end
    end
    
    df = DataFrame(results)
    mkpath("experiments/results")
    CSV.write("experiments/results/verification_report.csv", df)
    println("\n>>> Campaign Completed. Report saved to experiments/results/verification_report.csv")
end

master_campaign(20)
