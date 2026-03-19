# All-Algorithm Tier 3 Exhaustive Campaign
# Full 6-Algorithm Comparison under Stochastic Partial Observability
# Based on Manuscript Section 9.4 and D.1.3
# Author: Mohit Ranjan, 2026.
#
# Campaign Design:
#   - Seeds:      30 per configuration (for statistical power)
#   - Sigma_obs:  {0.01, 0.1, 0.5, 1.0}    — noise sweep
#   - Alpha (α):  {0.1, 0.5, 1.0}           — reflexive coupling
#   - Algorithms: 5 (PPO, SAC, EGP, FPRL, ICRL)
#
# Total: 30 seeds × 4 sigma × 3 alpha × 5 algos = 1800 runs
# Combined with the existing 2880-run dataset, total Tier 3 data = 5040+ rows

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using CSV, DataFrames, Statistics, Random, Dates, LinearAlgebra

const SEEDS      = 1:30
const SIGMA_LIST = [0.01, 0.1, 0.5, 1.0]
const ALPHA_LIST = [0.1, 0.5, 1.0]
const EPISODES   = 200
const STEPS_T3   = 150
const LR         = 3e-4
const OBS_DIM    = 2

total_configs = length(SEEDS) * length(SIGMA_LIST) * length(ALPHA_LIST) * 5
println("Total planned runs: $total_configs")   # 1800

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

function run_t3_all_algo(algo_type::String, sigma_obs::Float64, alpha::Float64, seed::Int)
    Random.seed!(seed)
    
    env    = Tier3Env(sigma_obs, 0.05, alpha)
    oracle = ReflexiveOracle(OBS_DIM)
    policy = GaussianPolicy(OBS_DIM)
    
    agent = if algo_type == "PPO"
        PPOAgent(policy, LR)
    elseif algo_type == "SAC"
        SACAgent(OBS_DIM, LR)
    elseif algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else # ICRL
        ICRLAgent(oracle, policy, LR, info_limit=0.5f0, beta=0.1f0)
    end
    
    ep_rewards = Float64[]
    ep_stabs   = Float64[]

    for ep in 1:EPISODES
        obs = ReflexiveRL.reset!(env)
        obs = Float32.(clamp.(obs, -5.0, 5.0))
        if any(isnan, obs); obs = zeros(Float32, OBS_DIM); end
        batch = []; ep_r = 0.0; ep_s = 0.0; prev_obs = copy(obs)

        for t in 1:STEPS_T3
            r_pred = 0.0f0
            if algo_type ∉ ["PPO", "SAC"]
                rr = agent.oracle(obs)[1]
                r_pred = (isnan(rr) || isinf(rr)) ? 0.0f0 : clamp(rr, -2f0, 2f0)
            end
            
            mu, sigma = agent.policy(obs)
            mu_v = isnan(mu[1]) ? 0.0f0 : mu[1]
            sg_v = (isnan(sigma[1]) || sigma[1] <= 0f0) ? 0.01f0 : sigma[1]
            a = clamp(mu_v + sg_v * randn(Float32), -2f0, 2f0)
            
            obs_next_raw = ReflexiveRL.step!(env, a, r_pred)
            obs_next = Float32.(clamp.(obs_next_raw, -5.0, 5.0))
            if any(isnan, obs_next); obs_next = zeros(Float32, OBS_DIM); end
            
            r = ReflexiveRL.reward(env, obs_next, a)
            r = isfinite(r) ? r : 0.0
            
            ep_r += r
            ep_s += norm(obs_next .- prev_obs)
            push!(batch, (obs, [a], [r_pred], obs_next, Float32(r), 0.0f0))
            prev_obs = copy(obs)
            obs = obs_next
        end
        
        if !isempty(batch) && algo_type ∉ ["SAC"]
            rl   = [Float64(b[5]) for b in batch]
            rtns = compute_advantages(rl)
            for i in 1:length(batch)
                batch[i] = (batch[i][1],batch[i][2],batch[i][3],batch[i][4],batch[i][5],Float32(rtns[i]))
            end
            if algo_type == "EGP"; update_egp!(agent, batch)
            elseif algo_type == "FPRL"
                t3p(S,A,R) = 0.9f0 .* S .+ 0.5f0 .* A .- Float32(alpha) .* R
                update_fprl!(agent, [b[1] for b in batch], t3p)
            elseif algo_type == "ICRL"; update_icrl!(agent, batch)
            else; update_ppo!(agent, batch)
            end
        end
        push!(ep_rewards, ep_r)
        push!(ep_stabs, ep_s / STEPS_T3)
    end

    last_n = max(1, length(ep_rewards) - 30)
    vr = filter(isfinite, ep_rewards[last_n:end]); vs = filter(isfinite, ep_stabs[last_n:end])
    return isempty(vr) ? -999.0 : mean(vr),
           (isempty(vr) || length(vr) < 2) ? 0.0 : std(vr),
           isempty(vs) ? 999.0 : mean(vs)
end

println("Starting All-Algo Tier 3 Campaign (~2160 runs)...")
run_counter = 0; start_time = now()

for sigma_obs in SIGMA_LIST, alpha in ALPHA_LIST, seed in SEEDS
    for algo in ["PPO", "SAC", "EGP", "FPRL", "ICRL"]
        global run_counter += 1
        avg_r, std_r, avg_s = run_t3_all_algo(algo, sigma_obs, alpha, seed)
        push!(results_df, (seed, algo, sigma_obs, alpha, avg_r, std_r, avg_s))
    end
    
    if run_counter % 60 == 0
        elapsed = now() - start_time
        pct = round(100 * run_counter / total_configs, digits=1)
        println("Progress: $run_counter/$total_configs ($pct%) | Elapsed: $elapsed")
        CSV.write("experiments/results/processed/tier3_all_algo_checkpoint.csv", results_df)
    end
end

CSV.write("experiments/results/processed/tier3_all_algo_full.csv", results_df)

summary_df = combine(groupby(results_df, [:sigma_obs, :algo]),
    :avg_reward => mean => :reward_mean,
    :avg_reward => std  => :reward_std,
    :stab_error => mean => :stab_mean
)

println("\n--- TIER 3 ALL-ALGORITHM SUMMARY ---")
println(summary_df)
CSV.write("experiments/results/processed/tier3_all_algo_summary.csv", summary_df)

elapsed = now() - start_time
println("\nTier 3 All-Algo Campaign Completed! Duration: $elapsed — Total runs: $(nrow(results_df))")
