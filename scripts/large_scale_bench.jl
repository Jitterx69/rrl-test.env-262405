# Large-Scale High-End Benchmark Campaign
# 2000 Runs per configuration to verify stability and find failure modes.
# Usage: julia -t auto scripts/large_scale_bench.jl

using Pkg
Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src"))

using ReflexiveRL
using Statistics, Random, CSV, DataFrames, Dates, Printf

# Explicitly import update functions to handle namespace shadowing
import ReflexiveRL: EGPAgent, FPRLAgent, PPOAgent, ICRLAgent, SACAgent
import ReflexiveRL: ReflexiveTrainer, reset!, step!, reward, compute_returns
import ReflexiveRL: update_egp!, update_fprl!, update_ppo!, update_icrl!, update_sac!
import ReflexiveRL: Tier1Env, Tier2Env, Tier3Env, ReflexiveOracle, GaussianPolicy

function run_single_experiment(algo_name, tier_idx, seed)
    Random.seed!(seed)
    
    env = if tier_idx == 1
        Tier1Env(1.0, 0.05)
    elseif tier_idx == 2
        Tier2Env(5, 0.5)
    else
        Tier3Env(0.1, 0.5)
    end
    
    state_dim = tier_idx == 3 ? 2 : 1
    
    agent = if algo_name == "EGP"
        EGPAgent(state_dim, 1)
    elseif algo_name == "FPRL"
        FPRLAgent(state_dim, 1)
    elseif algo_name == "PPO"
        PPOAgent(state_dim, 1)
    elseif algo_name == "ICRL"
        ICRLAgent(state_dim, 1)
    elseif algo_name == "SAC"
        SACAgent(state_dim, 1)
    else
        return nothing
    end
    
    config = Dict("epochs" => 50, "steps" => 100, "verbose" => false)
    trainer = ReflexiveTrainer(agent, env, config)
    
    train_quiet!(trainer)
    
    # 4. Eval
    eval_rewards = []
    s = reset!(env)
    for _ in 1:200
        r_pred = agent.oracle(s isa Vector ? s : [Float32(s)])[1]
        pol_input = vcat(s isa Vector ? s : [Float32(s)], [Float32(r_pred)])
        mu, sigma = agent.policy(pol_input)
        a = mu[1]
        s_next = step!(env, a, r_pred)
        push!(eval_rewards, reward(env, s_next, a))
        s = s_next
    end
    
    return mean(eval_rewards)
end

function train_quiet!(trainer)
    epochs = get(trainer.config, "epochs", 50)
    steps  = get(trainer.config, "steps", 100)
    
    for epoch in 1:epochs
        batch = []
        s = reset!(trainer.env)
        for t in 1:steps
            st_vec = s isa Vector ? s : [Float32(s)]
            r_pred = trainer.agent.oracle(st_vec)[1]
            pol_in = vcat(st_vec, [Float32(r_pred)])
            mu, sigma = trainer.agent.policy(pol_in)
            a = mu[1] + sigma[1] * randn(Float32)
            
            s_next = step!(trainer.env, a, r_pred)
            r = reward(trainer.env, s_next, a)
            
            sn_vec = s_next isa Vector ? s_next : [Float32(s_next)]
            push!(batch, (st_vec, [Float32(a)], [Float32(r_pred)], sn_vec, Float32(r), 0.0f0))
            s = s_next
        end
        
        rtns = compute_returns([b[5] for b in batch])
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(rtns[i]))
        end
        
        if trainer.agent isa EGPAgent
            update_egp!(trainer.agent, batch, trainer.env)
        elseif trainer.agent isa FPRLAgent
            update_fprl!(trainer.agent, batch, trainer.env)
        elseif trainer.agent isa PPOAgent
            update_ppo!(trainer.agent, batch)
        elseif trainer.agent isa ICRLAgent
            update_icrl!(trainer.agent, batch)
        elseif trainer.agent isa SACAgent
            update_sac!(trainer.agent, batch)
        end
    end
end

function run_campaign(n_runs=100)
    algos = ["EGP", "FPRL", "PPO", "ICRL", "SAC"]
    tiers = [1, 2, 3]
    results = []
    
    println(">>> Starting Sequential High-End Verification Campaign ($n_runs runs per config)")
    
    for tier in tiers
        for algo in algos
            print("Running Tier $tier - $algo... ")
            tier_results = zeros(n_runs)
            for i in 1:n_runs
                tier_results[i] = run_single_experiment(algo, tier, 1337 + i)
            end
            m = mean(tier_results)
            s = std(tier_results)
            @printf(" Mean: %.3f | Std: %.3f\n", m, s)
            push!(results, (tier=tier, algo=algo, mean=m, std=s))
        end
    end
    
    df = DataFrame(results)
    mkpath("experiments/results")
    CSV.write("experiments/results/verification_campaign.csv", df)
    println("\n>>> Campaign Completed.")
end

run_campaign(20)
