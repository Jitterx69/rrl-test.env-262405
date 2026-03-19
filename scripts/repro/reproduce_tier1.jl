# Reproduction Script: Tier 1 - Controlled Reflexive Dynamical System
# Based on Project Manuscript Draft I, Section 9.2 & D.1.1
# Author: Mohit Ranjan, 2026.
#
# This script executes a comprehensive comparative analysis of EGP, FPRL, 
# and a PPO baseline on the Tier 1 environment. We evaluate performance 
# across three levels of Reflexive Gain (alpha = 0.1, 0.5, 1.0) to 
# demonstrate how classical RL fails as reflexivity increases.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
import .ReflexiveRL: reset!, step!, reward
using CSV, DataFrames, Statistics, Random, Dates

# ---------------------------------------------------------
# 1. Experimental Configuration
# ---------------------------------------------------------

const ALPHAS = [0.1, 0.5, 1.0] # Reflexive gain levels
const SEEDS = 1:50              # Scaled to 50 seeds for "Most Accurate" baseline
const EPISODES = 200            # Total training episodes
const STEPS_PER_EP = 100       # Horizon (T)
const LR = 3e-4                # Learning rate shared across algorithms

# ---------------------------------------------------------
# 2. Result Collection Infrastructure
# ---------------------------------------------------------

results_df = DataFrame(
    alpha = Float64[],
    seed = Int[],
    algo = String[],
    avg_reward = Float64[],
    stability_error = Float64[]
)

mkpath("experiments/results/processed")

# ---------------------------------------------------------
# 3. Algorithm Execution Loop
# ---------------------------------------------------------

function run_experiment(algo_type::String, alpha::Float64, seed::Int)
    println(">>> Running $algo_type (alpha=$alpha, seed=$seed)...")
    Random.seed!(seed)
    
    # 3.1 Setup Environment and Networks
    env = Tier1Env(alpha)
    oracle = ReflexiveOracle(1)
    policy = GaussianPolicy(1)
    
    # 3.2 Initialize Selected Agent
    agent = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else # PPO Baseline
        PPOAgent(policy, LR)
    end
    
    total_rewards = []
    stability_errors = []

    # 3.3 Training Loop
    for ep in 1:EPISODES
        s = reset!(env)
        batch = []
        episode_reward = 0.0
        episode_stability = 0.0
        
        for t in 1:STEPS_PER_EP
            # Forward Pass through the reflexive loop
            # PPO doesn't use the oracle's signal for its policy
            r_pred = if algo_type == "PPO" 
                0.0 # Standard RL assumes zero predictive feedback
            else
                agent.oracle.model([Float32(s)])[1]
            end
            
            # Action Selection
            # For PPO/EGP/FPRL, the policy is conditioned on its input
            input_val = algo_type == "PPO" ? [Float32(s)] : [Float32(r_pred)]
            a_mean, a_std = agent.policy(input_val)
            a = a_mean[1] + a_std[1] * randn()
            
            # Environment Step
            # Note: The environment ALWAYS behaves reflexively (endogenous dynamics)
            # regardless of whether the agent accounts for it.
            s_next = step!(env, a, r_pred)
            r = reward(env, s_next, a)
            
            # Record transition
            # batch entry: (s_t, a_t, r_pred, s_t+1, reward, return_placeholder)
            push!(batch, ([Float32(s)], [Float32(a)], [Float32(r_pred)], [Float32(s_next)], r, 0.0))
            
            episode_reward += r
            episode_stability += abs(s - s_next)
            s = s_next
        end
        
        # 3.4 Return Computation and Advantage Estimation
        # (Using simple normalized returns for this comparison)
        rewards_list = [b[5] for b in batch]
        returns = compute_advantages(rewards_list)
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], returns[i])
        end
        
        # 3.5 Agent Optimization
        if algo_type == "EGP"
            update_egp!(agent, batch)
        elseif algo_type == "FPRL"
            # Define dynamics proxy for FPRL
            t1_model(S, A, R) = S .+ A .- Float32(env.α) .* R
            states_only = [b[1] for b in batch]
            update_fprl!(agent, states_only, t1_model)
        else
            update_ppo!(agent, batch)
        end
        
        push!(total_rewards, episode_reward)
        push!(stability_errors, episode_stability / STEPS_PER_EP)
    end
    
    return mean(total_rewards[end-50:end]), mean(stability_errors[end-50:end])
end

# ---------------------------------------------------------
# 4. Main Simulation Execution
# ---------------------------------------------------------

println("Starting Comparative Simulation Campaign...")

for alpha in ALPHAS
    for seed in SEEDS
        # 1. Run EGP (The proposed solution)
        egp_reward, egp_stab = run_experiment("EGP", alpha, seed)
        push!(results_df, (alpha, seed, "EGP", egp_reward, egp_stab))
        
        # 2. Run FPRL (The fixed-point variant)
        fprl_reward, fprl_stab = run_experiment("FPRL", alpha, seed)
        push!(results_df, (alpha, seed, "FPRL", fprl_reward, fprl_stab))
        
        # 3. Run PPO (The failing baseline)
        ppo_reward, ppo_stab = run_experiment("PPO", alpha, seed)
        push!(results_df, (alpha, seed, "PPO", ppo_reward, ppo_stab))
    end
end

# ---------------------------------------------------------
# 5. Data Export and Table Generation
# ---------------------------------------------------------

CSV.write("experiments/results/processed/tier1_full_comparison.csv", results_df)

# Group and summarize for the paper
summary_df = combine(groupby(results_df, [:alpha, :algo]), 
    :avg_reward => mean => :reward_mean,
    :avg_reward => std => :reward_std,
    :stability_error => mean => :stab_mean
)

println("\n--- TIER 1 SUMMARY RESULTS ---")
println(summary_df)

CSV.write("experiments/results/processed/tier1_summary_table.csv", summary_df)
println("\nComparative simulation completed. Data saved to experiments/results/processed/")
