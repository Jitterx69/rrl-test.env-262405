# Reproduction Script: Tier 3 - Stochastic Partial Observability
# Based on Project Manuscript Draft I, Section 9.4 & D.1.3
# Author: Mohit Ranjan, 2026.
#
# This campaign evaluates EGP, FPRL, and PPO in the hardest regime:
# the agent only receives NOISY observations of the hidden state.
# The reflexive oracle must predict system dynamics from corrupted sensors.
#
# Sweep: sigma_obs ∈ {0.01, 0.1, 0.5, 1.0} to map the "noise threshold"
# where classical RL fails but reflexive methods hold.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
import .ReflexiveRL: reset!, step!, reward
using CSV, DataFrames, Statistics, Random, Dates, LinearAlgebra

# ---------------------------------------------------------
# 1. Experimental Configuration
# ---------------------------------------------------------

# Sigma levels map to: Clean, Low-Noise, Medium-Noise, High-Noise
const SIGMA_OBS_LEVELS = [0.01, 0.1, 0.5, 1.0]  # Observation noise sweep
const SEEDS = 1:10
const EPISODES = 200
const STEPS_PER_EP = 150       # T = 150 as per Table E.3
const LR = 3e-4

# ---------------------------------------------------------
# 2. Adapted Architectures for Partial Observability
# In Tier 3, the policy and oracle take 2D observations as input
# ---------------------------------------------------------

const OBS_DIM = 2  # Observation dimension (= state dim)

# ---------------------------------------------------------
# 3. Result Collection
# ---------------------------------------------------------

results_df = DataFrame(
    seed        = Int[],
    algo        = String[],
    sigma_obs   = Float64[],
    avg_reward  = Float64[],
    stab_error  = Float64[]  # E_stab = ||o_t - o_{t-1}||
)

mkpath("experiments/results/processed")

# ---------------------------------------------------------
# 4. Experiment Runner
# ---------------------------------------------------------

function run_tier3_experiment(algo_type::String, sigma_obs::Float64, seed::Int)
    println(">>> Running Tier 3: $algo_type (sigma_obs=$sigma_obs, seed=$seed)...")
    Random.seed!(seed)
    
    env = Tier3Env(sigma_obs, 0.05, 0.5)      # Moderate reflexive coupling
    
    # Tier 3 needs 2D input for oracle and policy
    oracle  = ReflexiveOracle(OBS_DIM)         # Observes 2D noisy state
    policy  = GaussianPolicy(OBS_DIM)          # Acts on 2D noisy state
    
    agent = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else
        PPOAgent(policy, LR)
    end
    
    all_ep_rewards = Float64[]
    all_stab_errors = Float64[]

    for ep in 1:EPISODES
        obs = ReflexiveRL.reset!(env)
        # Ensure obs is a clean Float32 vector
        obs = Float32.(clamp.(obs, -5.0, 5.0))
        if any(isnan, obs); obs = zeros(Float32, OBS_DIM); end

        batch = []
        ep_reward  = 0.0
        ep_stab   = 0.0
        prev_obs = copy(obs)

        for t in 1:STEPS_PER_EP
            # 1. Oracle prediction from noisy observation
            rho_raw = algo_type == "PPO" ? zeros(Float32, 1) : oracle(obs)
            r_pred_val = isnan(rho_raw[1]) ? 0.0f0 : clamp(rho_raw[1], -2.0f0, 2.0f0)
            
            # 2. Policy acts on 2D noisy observation (same for all algo types)
            # The oracle signal rho is used ONLY in the environment dynamics,
            # not as the policy input (Section 5.1: policy conditions on observation).
            mu, sigma = policy(obs)
            mu_safe    = isnan(mu[1])    ? 0.0f0 : mu[1]
            sigma_safe = (isnan(sigma[1]) || sigma[1] <= 0f0) ? 0.01f0 : sigma[1]
            
            a = clamp(mu_safe + sigma_safe * randn(Float32), -2.0f0, 2.0f0)
            
            # 3. Step environment — returns NOISY observation
            obs_next_raw = ReflexiveRL.step!(env, a, r_pred_val)
            obs_next = Float32.(clamp.(obs_next_raw, -5.0, 5.0))
            if any(isnan, obs_next); obs_next = zeros(Float32, OBS_DIM); end
            
            # 4. Reward uses noisy observation (agent cannot see truth)
            r = ReflexiveRL.reward(env, obs_next, a)
            r = isfinite(r) ? r : 0.0
            
            # 5. E_stab: consecutive observation distance (Section 10.2)
            ep_stab += norm(obs_next .- prev_obs)
            ep_reward += r
            
            push!(batch, (obs, [a], [r_pred_val], obs_next, Float32(r), 0.0f0))
            
            prev_obs = copy(obs)
            obs = obs_next
        end
        
        # Training step
        if !isempty(batch)
            rewards_list = [Float64(b[5]) for b in batch]
            returns = compute_advantages(rewards_list)
            for i in 1:length(batch)
                batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(returns[i]))
            end
            
            if algo_type == "EGP"; update_egp!(agent, batch)
            elseif algo_type == "FPRL"
                # Dynamics proxy for the 2D linear system
                # s_{t+1} ≈ 0.9 * s_t + 0.5 * a - alpha * rho
                t3_model(S, A, R) = 0.9f0 .* S .+ 0.5f0 .* A .- 0.5f0 .* R
                update_fprl!(agent, [b[1] for b in batch], t3_model)
            else; update_ppo!(agent, batch)
            end
        end
        
        push!(all_ep_rewards, ep_reward)
        push!(all_stab_errors, ep_stab / STEPS_PER_EP)
    end
    
    last_n = max(1, length(all_ep_rewards) - 30)
    valid_rewards = filter(isfinite, all_ep_rewards[last_n:end])
    valid_stabs   = filter(isfinite, all_stab_errors[last_n:end])
    
    return isempty(valid_rewards) ? -999.0 : mean(valid_rewards),
           isempty(valid_stabs)   ? 999.0  : mean(valid_stabs)
end

# ---------------------------------------------------------
# 5. Main Campaign Execution
# ---------------------------------------------------------

println("Starting Tier 3 Stochastic Observability Campaign...")
start_time = now()

for sigma_obs in SIGMA_OBS_LEVELS
    println("\n=== Observation Noise: sigma_obs = $sigma_obs ===")
    for seed in SEEDS
        for algo in ["PPO", "EGP", "FPRL"]
            avg_rew, avg_stab = run_tier3_experiment(algo, sigma_obs, seed)
            push!(results_df, (seed, algo, sigma_obs, avg_rew, avg_stab))
        end
    end
end

# ---------------------------------------------------------
# 6. Save & Summary
# ---------------------------------------------------------

CSV.write("experiments/results/processed/tier3_full_comparison.csv", results_df)

summary_df = combine(groupby(results_df, [:sigma_obs, :algo]),
    :avg_reward => mean => :reward_mean,
    :stab_error => mean => :stab_mean
)

println("\n--- TIER 3 SUMMARY RESULTS ---")
println(summary_df)

CSV.write("experiments/results/processed/tier3_summary_table.csv", summary_df)

elapsed = now() - start_time
println("\nTier 3 Simulation Completed. Duration: $elapsed")
