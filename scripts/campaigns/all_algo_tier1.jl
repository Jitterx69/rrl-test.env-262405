# All-Algorithm Tier 1 Exhaustive Campaign
# Full 5-Algorithm Comparison: PPO, SAC, EGP, FPRL, ICRL
# Based on Manuscript Section 9.1 and D.1.1
# Author: Mohit Ranjan, 2026.
#
# Campaign Design:
#   - Seeds:    50 per configuration
#   - Alpha (α): {0.1, 0.5, 1.0}  — reflexive gain sweep
#   - Algorithms: 5 (PPO, SAC, EGP, FPRL, ICRL)
#
# Total: 50 seeds × 3 alpha × 5 algos = 750 runs
# Combined with the previous 50-seed run, total Tier 1 data = 1800+ rows

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using CSV, DataFrames, Statistics, Random, Dates

const SEEDS      = 1:50
const ALPHA_LIST = [0.1, 0.5, 1.0]
const EPISODES   = 300
const STEPS_T1   = 200
const LR         = 3e-4

total_configs = length(SEEDS) * length(ALPHA_LIST) * 5
println("Total planned runs: $total_configs")   # 750

results_df = DataFrame(
    seed       = Int[],
    algo       = String[],
    alpha      = Float64[],
    avg_reward = Float64[],
    reward_std = Float64[],
    stab_error = Float64[]
)

mkpath("experiments/results/processed")

function run_t1_all_algo(algo_type::String, alpha::Float64, seed::Int)
    Random.seed!(seed)
    
    env    = Tier1Env(alpha, 0.05)
    oracle = ReflexiveOracle(1)
    policy = GaussianPolicy(1)
    
    agent = if algo_type == "PPO"
        PPOAgent(policy, LR)
    elseif algo_type == "SAC"
        SACAgent(1, LR)
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
        s = ReflexiveRL.reset!(env)
        if isnan(s) || isinf(s); s = 0.0; end
        batch = []; ep_r = 0.0; ep_s = 0.0

        for t in 1:STEPS_T1
            r_pred = 0.0f0
            if algo_type ∉ ["PPO", "SAC"]
                rr = agent.oracle.model([Float32(s)])[1]
                r_pred = (isnan(rr) || isinf(rr)) ? 0.0f0 : clamp(rr, -2f0, 2f0)
            end
            
            mu, sigma = agent.policy([Float32(s)])
            mu_v = isnan(mu[1]) ? 0.0f0 : mu[1]
            sg_v = (isnan(sigma[1]) || sigma[1] <= 0f0) ? 0.01f0 : sigma[1]
            a = clamp(mu_v + sg_v * randn(Float32), -1.0f0, 1.0f0)
            
            s_next = ReflexiveRL.step!(env, a, r_pred)
            if isnan(s_next) || isinf(s_next); s_next = 0.0; end
            
            r = ReflexiveRL.reward(env, s_next, a)
            r = isfinite(r) ? r : 0.0
            
            ep_r += r
            ep_s += abs(s_next - s)
            push!(batch, ([Float32(s)], [a], [r_pred], [Float32(s_next)], Float32(r), 0.0f0))
            s = s_next
        end
        
        if !isempty(batch) && algo_type ∉ ["SAC"]
            rl = [Float64(b[5]) for b in batch]
            rtns = compute_advantages(rl)
            for i in 1:length(batch)
                batch[i] = (batch[i][1],batch[i][2],batch[i][3],batch[i][4],batch[i][5],Float32(rtns[i]))
            end
            if algo_type == "EGP"
                update_egp!(agent, batch)
            elseif algo_type == "FPRL"
                t1_proxy(S,A,R) = S .+ A .- Float32(alpha) .* R
                update_fprl!(agent, [b[1] for b in batch], t1_proxy)
            elseif algo_type == "ICRL"
                update_icrl!(agent, batch)
            else
                update_ppo!(agent, batch)
            end
        end
        
        push!(ep_rewards, ep_r)
        push!(ep_stabs, ep_s / STEPS_T1)
    end
    
    last_n = max(1, length(ep_rewards) - 50)
    vr = filter(isfinite, ep_rewards[last_n:end])
    vs = filter(isfinite, ep_stabs[last_n:end])
    return isempty(vr) ? -999.0 : mean(vr),
           (isempty(vr) || length(vr) < 2) ? 0.0 : std(vr),
           isempty(vs) ? 999.0 : mean(vs)
end

println("Starting All-Algo Tier 1 Campaign (~900 runs)...")
run_counter = 0
start_time  = now()

for alpha in ALPHA_LIST, seed in SEEDS
    for algo in ["PPO", "SAC", "EGP", "FPRL", "ICRL"]
        global run_counter += 1
        avg_r, std_r, avg_s = run_t1_all_algo(algo, alpha, seed)
        push!(results_df, (seed, algo, alpha, avg_r, std_r, avg_s))
    end
    
    if run_counter % 60 == 0
        elapsed = now() - start_time
        pct = round(100 * run_counter / total_configs, digits=1)
        println("Progress: $run_counter/$total_configs ($pct%) | Elapsed: $elapsed")
        CSV.write("experiments/results/processed/tier1_all_algo_checkpoint.csv", results_df)
    end
end

CSV.write("experiments/results/processed/tier1_all_algo_full.csv", results_df)

summary_df = combine(groupby(results_df, [:alpha, :algo]),
    :avg_reward => mean => :reward_mean,
    :avg_reward => std  => :reward_std,
    :stab_error => mean => :stab_mean
)

println("\n--- TIER 1 ALL-ALGORITHM SUMMARY ---")
println(summary_df)
CSV.write("experiments/results/processed/tier1_all_algo_summary.csv", summary_df)

elapsed = now() - start_time
println("\nTier 1 All-Algo Campaign Completed! Duration: $elapsed — Total runs: $(nrow(results_df))")
