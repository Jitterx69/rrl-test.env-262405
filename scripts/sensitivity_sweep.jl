# Research-Grade Sensitivity Sweep: Reflexive Gain Analysis
# Based on Project Manuscript Draft I, Section G.6 & 11.7
# Author: Mohit Ranjan, 2026.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
import .ReflexiveRL: reset!, step!, reward
using CSV, DataFrames, Statistics, Random, Dates, Printf

# ---------------------------------------------------------
# 1. Sweep Configuration
# ---------------------------------------------------------

const ALPHA_RANGE = 0.0:0.1:1.5    
const SEEDS_PER_POINT = 3
const EPISODES = 50
const STEPS_PER_EP = 100
const LR = 3e-4

# ---------------------------------------------------------
# 2. Result Collection
# ---------------------------------------------------------

sweep_results = DataFrame(
    alpha = Float64[],
    seed = Int[],
    algo = String[],
    final_reward = Float64[],
    final_stability = Float64[]
)

mkpath("experiments/results/processed")

# ---------------------------------------------------------
# 3. Modular Experiment Runner
# ---------------------------------------------------------

function run_point(algo_type::String, alpha::Float64, seed::Int)
    Random.seed!(seed)
    
    env = Tier1Env(alpha)
    oracle = ReflexiveOracle(1, 1)
    policy = GaussianPolicy(2, 1) 
    
    agent = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else 
        PPOAgent(oracle, policy, LR)
    end
    
    rewards = []
    stabilities = []

    for ep in 1:EPISODES
        s = reset!(env)
        batch = []
        ep_reward = 0.0
        ep_stab = 0.0
        
        for t in 1:STEPS_PER_EP
            r_pred = algo_type == "PPO" ? 0.0f0 : agent.oracle([Float32(s)])[1]
            input_val = [Float32(s), Float32(r_pred)]
            
            a_mean, a_std = agent.policy(input_val)
            a = a_mean[1] + a_std[1] * randn(Float32)
            
            s_next = step!(env, a, r_pred)
            r = reward(env, s_next, a)
            
            push!(batch, ([Float32(s)], [Float32(a)], [Float32(r_pred)], [Float32(s_next)], Float32(r), 0.0f0))
            
            ep_reward += r
            ep_stab += abs(s - s_next)
            s = s_next
        end
        
        # Train
        returns = compute_advantages([b[5] for b in batch])
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(returns[i]))
        end
        
        if algo_type == "EGP"; update_egp!(agent, batch, env)
        elseif algo_type == "FPRL"
            update_fprl!(agent, batch, env)
        else; update_ppo!(agent, batch); end
        
        push!(rewards, ep_reward)
        push!(stabilities, ep_stab / STEPS_PER_EP)
    end
    
    return mean(rewards[end-10:end]), mean(stabilities[end-10:end])
end

# ---------------------------------------------------------
# 4. Campaign Execution
# ---------------------------------------------------------

println("Starting Final Sensitivity Sweep...")
start_time = now()

for alpha in ALPHA_RANGE
    @printf("Testing Alpha: %.2f ... ", alpha)
    for seed in 1:SEEDS_PER_POINT
        for algo in ["PPO", "EGP", "FPRL"]
            reward_val, stab_val = run_point(algo, alpha, seed)
            push!(sweep_results, (alpha, seed, algo, reward_val, stab_val))
        end
    end
    println("Done.")
end

# ---------------------------------------------------------
# 5. Export
# ---------------------------------------------------------

CSV.write("experiments/results/processed/alpha_sensitivity_sweep.csv", sweep_results)
duration = now() - start_time
println("\nSweep Complete! Duration: $duration")
