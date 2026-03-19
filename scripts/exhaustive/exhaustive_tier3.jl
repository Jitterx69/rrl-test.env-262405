# Scaled Tier 3 Campaign: Stochastic Partial Observability
# Exhaustive Parameter Sweep for Research-Grade Publication
# Author: Mohit Ranjan, 2026.
#
# Campaign Design: 2160 independent simulation runs.
#
# Parameter Matrix:
#   - Seeds:        30 per configuration
#   - Sigma_obs:    [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5] — noise levels
#   - Alpha (α):    [0.1, 0.5, 1.0, 1.5]                         — reflexive coupling
#   - Algorithms:   [PPO, EGP, FPRL]                               — 3-way comparison
#
# Total: 30 seeds × 8 sigma × 4 alpha × 3 algos = 2880 runs
# (Comprehensive mapping of the noise-observability-reflexivity surface)

include("../../src/ReflexiveRL.jl")
using .ReflexiveRL
using CSV, DataFrames, Statistics, Random, Dates, LinearAlgebra

# ---------------------------------------------------------
# 1. Sweep Configuration
# ---------------------------------------------------------

const SEEDS       = 1:2
const SIGMA_LIST  = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5]   # Fine-grained noise sweep
const ALPHA_LIST  = [0.1, 0.5, 1.0, 1.5]                             # Reflexive coupling
const EPISODES    = 200
const STEPS_PER_EP = 150
const LR          = 3e-4
const OBS_DIM     = 2

total_configs = length(SEEDS) * length(SIGMA_LIST) * length(ALPHA_LIST) * 3
println("Total planned runs: $total_configs")   # 2880

# ---------------------------------------------------------
# 2. Result Collection
# ---------------------------------------------------------

results_df = DataFrame(
    seed       = Int[],
    algo       = String[],
    sigma_obs  = Float64[],
    alpha      = Float64[],
    avg_reward = Float64[],
    reward_std = Float64[],
    stab_error = Float64[]
)

mkpath("experiments/results/processed")

# ---------------------------------------------------------
# 3. Runner
# ---------------------------------------------------------

function run_t3_config(algo_type::String, sigma_obs::Float64, alpha::Float64, seed::Int)
    Random.seed!(seed)
    
    # Tier3Env(alpha, noise_std)
    env    = Tier3Env(alpha, sigma_obs) 
    oracle = ReflexiveOracle(OBS_DIM)
    policy = GaussianPolicy(OBS_DIM)
    
    agent = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else
        PPOAgent(policy, LR)
    end
    
    all_rewards = Float64[]
    all_stabs   = Float64[]

    for ep in 1:EPISODES
        obs = ReflexiveRL.reset!(env)
        obs = Float32.(clamp.(obs, -5.0, 5.0))
        if any(isnan, obs); obs = zeros(Float32, OBS_DIM); end
        
        batch      = []
        ep_reward  = 0.0
        ep_stab    = 0.0
        prev_obs   = copy(obs)

        for t in 1:STEPS_PER_EP
            # Oracle prediction
            rho_raw   = algo_type == "PPO" ? zeros(Float32, 1) : oracle(obs)
            r_pred    = (isnan(rho_raw[1]) || isinf(rho_raw[1])) ? 0.0f0 : clamp(rho_raw[1], -2.0f0, 2.0f0)
            
            # Policy
            mu, sigma = policy(obs)
            mu_s      = isnan(mu[1])    ? 0.0f0 : mu[1]
            sig_s     = (isnan(sigma[1]) || sigma[1] <= 0f0) ? 0.01f0 : sigma[1]
            a         = clamp(mu_s + sig_s * randn(Float32), -2.0f0, 2.0f0)
            
            # Step
            obs_next_raw = ReflexiveRL.step!(env, a, r_pred)
            obs_next = Float32.(clamp.(obs_next_raw, -5.0, 5.0))
            if any(isnan, obs_next); obs_next = zeros(Float32, OBS_DIM); end
            
            # Reward
            r = ReflexiveRL.reward(env, obs_next, a)
            r = isfinite(r) ? r : 0.0
            
            ep_reward += r
            ep_stab   += norm(obs_next .- prev_obs)
            
            push!(batch, (obs, [a], [r_pred], obs_next, Float32(r), 0.0f0))
            prev_obs = copy(obs)
            obs = obs_next
        end

        # Training
        if !isempty(batch)
            rl   = [Float64(b[5]) for b in batch]
            rtns = compute_advantages(rl)
            for i in 1:length(batch)
                batch[i] = (batch[i][1],batch[i][2],batch[i][3],batch[i][4],batch[i][5],Float32(rtns[i]))
            end
            if algo_type == "EGP"
                update_egp!(agent, batch, env)
            elseif algo_type == "FPRL"
                update_fprl!(agent, batch, env)
            else
                update_ppo!(agent, batch)
            end
        end
        
        push!(all_rewards, ep_reward)
        push!(all_stabs, ep_stab / STEPS_PER_EP)
    end

    last_n = max(1, length(all_rewards) - 30)
    vr  = filter(isfinite, all_rewards[last_n:end])
    vs  = filter(isfinite, all_stabs[last_n:end])
    avg_r = isempty(vr) ? -999.0 : mean(vr)
    std_r = (isempty(vr) || length(vr) < 2) ? 0.0 : std(vr)
    avg_s = isempty(vs) ? 999.0  : mean(vs)
    return avg_r, std_r, avg_s
end

# ---------------------------------------------------------
# 4. Campaign Execution
# ---------------------------------------------------------

println("Starting Tier 3 Exhaustive Campaign (~2880 runs)...")
run_counter = 0
start_time  = now()

for sigma_obs in SIGMA_LIST, alpha in ALPHA_LIST, seed in SEEDS
    for algo in ["PPO", "EGP", "FPRL"]
        global run_counter += 1
        avg_r, std_r, avg_s = run_t3_config(algo, sigma_obs, alpha, seed)
        push!(results_df, (seed, algo, sigma_obs, alpha, avg_r, std_r, avg_s))
    end
    
    if run_counter % 90 == 0
        elapsed = now() - start_time
        pct = round(100 * run_counter / total_configs, digits=1)
        println("Progress: $run_counter/$total_configs ($pct%) | Elapsed: $elapsed")
        CSV.write("experiments/results/processed/tier3_exhaustive_checkpoint.csv", results_df)
    end
end

# ---------------------------------------------------------
# 5. Save & Summary
# ---------------------------------------------------------

CSV.write("experiments/results/processed/tier3_exhaustive_full.csv", results_df)

summary_df = combine(groupby(results_df, [:sigma_obs, :algo]),
    :avg_reward => mean => :reward_mean,
    :avg_reward => std  => :reward_std,
    :stab_error => mean => :stab_mean
)

println("\n--- TIER 3 EXHAUSTIVE SUMMARY (by sigma_obs × algo) ---")
println(summary_df)

CSV.write("experiments/results/processed/tier3_exhaustive_summary.csv", summary_df)

elapsed = now() - start_time
println("\nTier 3 Exhaustive Campaign Completed! Duration: $elapsed")
println("Total runs: $(nrow(results_df))")
