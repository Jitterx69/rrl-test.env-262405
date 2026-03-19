# Research-Grade Sensitivity Sweep: Reflexive Gain Analysis
# Based on Project Manuscript Draft I, Section G.6 & 11.7
# Author: Mohit Ranjan, 2026.
#
# This script performs a high-resolution sweep of the reflexive gain parameter (alpha) 
# from 0.0 to 1.5 in increments of 0.05. It is designed to identify the bifurcation 
# point where classical RL (PPO) fails and analyze the robustness of EGP and FPRL.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
import .ReflexiveRL: reset!, step!, reward
using CSV, DataFrames, Statistics, Random, Dates

# ---------------------------------------------------------
# 1. Sweep Configuration
# ---------------------------------------------------------

const ALPHA_RANGE = 0.0:0.05:1.5  # High-resolution sweep
const SEEDS_PER_POINT = 10        # Sufficient for sensitivity mapping
const EPISODES = 100              # Focus on convergence speed and stability
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
    oracle = ReflexiveOracle(1)
    policy = GaussianPolicy(1)
    
    agent = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else 
        PPOAgent(policy, LR)
    end
    
    rewards = []
    stabilities = []

    for ep in 1:EPISODES
        s = reset!(env)
        batch = []
        ep_reward = 0.0
        ep_stab = 0.0
        
        for t in 1:STEPS_PER_EP
            r_pred = algo_type == "PPO" ? 0.0 : agent.oracle.model([Float32(s)])[1]
            input_val = algo_type == "PPO" ? [Float32(s)] : [Float32(r_pred)]
            
            a_mean, a_std = agent.policy(input_val)
            a = a_mean[1] + a_std[1] * randn()
            
            s_next = step!(env, a, r_pred)
            r = reward(env, s_next, a)
            
            push!(batch, ([Float32(s)], [Float32(a)], [Float32(r_pred)], [Float32(s_next)], r, 0.0))
            
            ep_reward += r
            ep_stab += abs(s - s_next)
            s = s_next
        end
        
        # Train
        returns = compute_advantages([b[5] for b in batch])
        for i in 1:length(batch); batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], returns[i]); end
        
        if algo_type == "EGP"; update_egp!(agent, batch)
        elseif algo_type == "FPRL"
            t1_model(S, A, R) = S .+ A .- Float32(env.α) .* R
            update_fprl!(agent, [b[1] for b in batch], t1_model)
        else; update_ppo!(agent, batch); end
        
        push!(rewards, ep_reward)
        push!(stabilities, ep_stab / STEPS_PER_EP)
    end
    
    return mean(rewards[end-20:end]), mean(stabilities[end-20:end])
end

# ---------------------------------------------------------
# 4. Campaign Execution
# ---------------------------------------------------------

println("Starting High-Resolution Sensitivity Sweep (Alpha 0.0 -> 1.5)...")
start_time = now()

for alpha in ALPHA_RANGE
    print("Testing Alpha: $alpha ... ")
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

# Generate Summary Report
summary_df = combine(groupby(sweep_results, [:alpha, :algo]), 
    :final_reward => mean => :reward_mean,
    :final_stability => mean => :stab_mean
)
CSV.write("experiments/results/processed/alpha_sensitivity_summary.csv", summary_df)

duration = now() - start_time
println("\nSweep Complete! Duration: $duration")
println("Results saved to: experiments/results/processed/alpha_sensitivity_sweep.csv")
