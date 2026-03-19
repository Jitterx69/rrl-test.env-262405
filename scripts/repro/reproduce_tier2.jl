# Reproduction Script: Tier 2 - Multi-Agent Resource Allocation
# Based on Project Manuscript Draft I, Section 9.3 & D.1.2
# Author: Mohit Ranjan, 2026.
#
# This campaign evaluates coordination under shared reflexive feedback. 
# We analyze how different algorithms handle the "Adversarial Collapse" 
# threshold where individual incentives for consumption lead to 
# systemic resource depletion.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using CSV, DataFrames, Statistics, Random, Dates

# ---------------------------------------------------------
# 1. Experimental Configuration
# ---------------------------------------------------------

const N_AGENTS = 5               # Five agents as per D.1.2
const SEEDS = 1:10               # Multi-seed for accurate mean
const EPISODES = 300             # Tier 2 requires more complex coordination
const STEPS_PER_EP = 200        # Horizon defined in E.5
const LR = 3e-4

# ---------------------------------------------------------
# 2. Result Collection
# ---------------------------------------------------------

results_df = DataFrame(
    seed = Int[],
    algo = String[],
    avg_reward = Float64[],
    collapse_count = Int[],
    final_resource_level = Float64[]
)

mkpath("experiments/results/processed")

# ---------------------------------------------------------
# 3. Multi-Agent Runner
# ---------------------------------------------------------

function run_tier2_experiment(algo_type::String, seed::Int)
    println(">>> Running Tier 2: $algo_type (seed=$seed)...")
    Random.seed!(seed)
    
    env = Tier2Env(N_AGENTS)
    oracle = ReflexiveOracle(1)
    policy = GaussianPolicy(1)
    
    agent_model = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else 
        PPOAgent(policy, LR)
    end
    
    total_rewards = []
    resource_levels = []
    collapses = 0

    for ep in 1:EPISODES
        s = ReflexiveRL.reset!(env)
        # Guard: ensure starting state is finite
        if isnan(s) || isinf(s); s = 0.5; end

        batch = []
        episode_reward = 0.0
        
        for t in 1:STEPS_PER_EP
            # 1. Shared Oracle Prediction
            r_pred_raw = algo_type == "PPO" ? 0.0f0 : agent_model.oracle.model([Float32(s)])[1]
            # Guard: NaN oracle output defaults to 0
            r_pred = isnan(r_pred_raw) || isinf(r_pred_raw) ? 0.0f0 : r_pred_raw
            
            # 2. Independent Actions for N Agents
            actions = Float32[]
            input_val = algo_type == "PPO" ? [Float32(s)] : [Float32(r_pred)]
            
            # Agents share the same policy network (Parameter Sharing)
            mu, sigma = agent_model.policy(input_val)
            # Guard: NaN policy outputs default to safe values
            mu_val = isnan(mu[1]) ? 0.0f0 : mu[1]
            sigma_val = (isnan(sigma[1]) || sigma[1] <= 0.0f0) ? 0.01f0 : sigma[1]
            
            for i in 1:N_AGENTS
                a_i = mu_val + sigma_val * randn(Float32)
                push!(actions, clamp(a_i, 0.0f0, Float32(env.a_max)))
            end
            
            # 3. Environment Step
            s_next = ReflexiveRL.step!(env, actions, r_pred)
            if isnan(s_next) || isinf(s_next); s_next = 0.0; end

            # 4. Per-agent rewards and episode accumulation
            rewards = [ReflexiveRL.reward(env, s_next, actions[i]) for i in 1:N_AGENTS]
            step_reward = mean(filter(isfinite, rewards))
            episode_reward += isnan(step_reward) ? 0.0 : step_reward
            
            # Record if collapse occurred  
            if s_next < 0.05
                collapses += 1
                break # Early termination as per E.3.5
            end
            
            # Record for training (using mean action/reward for the shared model)
            push!(batch, ([Float32(s)], [Float32(mean(actions))], [Float32(r_pred)], [Float32(s_next)], Float32(step_reward), 0.0f0))
            
            s = s_next
        end
        
        # 5. Training Step
        if !isempty(batch)
            rewards_list = [b[5] for b in batch]
            returns = compute_advantages(rewards_list)
            for i in 1:length(batch); batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], returns[i]); end
            
            if algo_type == "EGP"; update_egp!(agent_model, batch)
            elseif algo_type == "FPRL"
                # Dynamics proxy for Tier 2 resource allocation
                t2_model(S, A, R) = S .- (Float32(N_AGENTS) .* A) .+ 0.5f0 .* S .* (1.0f0 .- S)
                update_fprl!(agent_model, [b[1] for b in batch], t2_model)
            else; update_ppo!(agent_model, batch); end
        end
        
        push!(total_rewards, episode_reward)
    end
    
    # Take mean of last 50 episodes, filtering out any residual NaN
    final_rewards = filter(isfinite, total_rewards[end-50:end])
    avg_rew = isempty(final_rewards) ? -999.0 : mean(final_rewards)
    return avg_rew, collapses, env.state
end

# ---------------------------------------------------------
# 4. Main Campaign
# ---------------------------------------------------------

println("Starting Tier 2 Comparative Campaign...")

for seed in SEEDS
    for algo in ["EGP", "FPRL", "PPO"]
        avg_rew, coll, final_s = run_tier2_experiment(algo, seed)
        push!(results_df, (seed, algo, avg_rew, coll, final_s))
    end
end

# ---------------------------------------------------------
# 5. Summary and Save
# ---------------------------------------------------------

CSV.write("experiments/results/processed/tier2_full_comparison.csv", results_df)

summary_df = combine(groupby(results_df, :algo), 
    :avg_reward => mean => :reward_mean,
    :collapse_count => sum => :total_collapses,
    :final_resource_level => mean => :final_level_mean
)

println("\n--- TIER 2 SUMMARY RESULTS ---")
println(summary_df)

CSV.write("experiments/results/processed/tier2_summary_table.csv", summary_df)
println("\nTier 2 Simulation Completed.")
