# All-Algorithm Tier 2 Exhaustive Campaign
# Full 5-Algorithm Comparison: PPO, SAC, EGP, FPRL, ICRL
# Based on Manuscript Section 9.3 and D.1.2
# Author: Mohit Ranjan, 2026.
#
# Campaign Design:
#   - Seeds:      50 per configuration
#   - N_Agents:   {3, 5, 10}     — scalability test
#   - Algorithms: 5              — all manuscript algorithms
#
# Total: 50 seeds × 3 N × 5 algos = 750 runs (exhaustive, 1 beta/gamma at canonical values)
#
# NOTE: SAC and ICRL are single-agent; they decide for all agents equally.
# MADDPG/QMIX (true centralized multi-agent) are out of scope of this framework.

include("../../src/ReflexiveRL.jl")
using .ReflexiveRL
using CSV, DataFrames, Statistics, Random, Dates

const SEEDS         = 1:50
const N_AGENTS_LIST = [3, 5, 10]
const EPISODES      = 300
const STEPS_T2      = 200
const LR            = 3e-4

# Canonical Tier 2 parameters (Appendix D.1.2)
const BETA_CANON   = 0.5
const GAMMA_CANON  = 1.0

total_configs = length(SEEDS) * length(N_AGENTS_LIST) * 5
println("Total planned runs: $total_configs")   # 750

results_df = DataFrame(
    seed              = Int[],
    algo              = String[],
    n_agents          = Int[],
    avg_reward        = Float64[],
    reward_std        = Float64[],
    collapse_count    = Int[],
    final_resource    = Float64[]
)

mkpath("experiments/results/processed")

function run_t2_all_algo(algo_type::String, n_agents::Int, seed::Int)
    Random.seed!(seed)
    
    env    = Tier2Env(n_agents, BETA_CANON, 0.5, GAMMA_CANON)
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
    collapses = 0

    for ep in 1:EPISODES
        s = ReflexiveRL.reset!(env)
        if isnan(s) || isinf(s); s = 0.5; end
        batch = []; ep_r = 0.0

        for t in 1:STEPS_T2
            r_pred = 0.0f0
            if algo_type ∉ ["PPO", "SAC"]
                rr = agent.oracle.model([Float32(s)])[1]
                r_pred = (isnan(rr) || isinf(rr)) ? 0.0f0 : clamp(rr, -2f0, 2f0)
            end
            
            mu, sigma = agent.policy([Float32(s)])
            mu_v = isnan(mu[1]) ? 0.0f0 : mu[1]
            sg_v = (isnan(sigma[1]) || sigma[1] <= 0f0) ? 0.01f0 : sigma[1]
            
            actions = Float32[clamp(mu_v + sg_v * randn(Float32), 0f0, Float32(env.a_max)) for _ in 1:n_agents]

            s_next = ReflexiveRL.step!(env, actions, r_pred)
            if isnan(s_next) || isinf(s_next); s_next = 0.0; end

            rewards = [ReflexiveRL.reward(env, s_next, Float64(actions[i])) for i in 1:n_agents]
            step_r  = isempty(filter(isfinite, rewards)) ? 0.0 : mean(filter(isfinite, rewards))
            ep_r += step_r

            if s_next < 0.05
                collapses += 1
                break
            end

            push!(batch, ([Float32(s)], [Float32(mean(actions))], [r_pred], [Float32(s_next)], Float32(step_r), 0.0f0))
            s = s_next
        end
        
        if !isempty(batch) && algo_type ∉ ["SAC"]
            rl   = [Float64(b[5]) for b in batch]
            rtns = compute_advantages(rl)
            for i in 1:length(batch)
                batch[i] = (batch[i][1],batch[i][2],batch[i][3],batch[i][4],batch[i][5],Float32(rtns[i]))
            end
            if algo_type == "EGP"; update_egp!(agent, batch, env)
            elseif algo_type == "FPRL"
                t2p(S,A,R) = S .- (Float32(n_agents) .* A) .+ Float32(BETA_CANON) .* S .* (1f0 .- S)
                update_fprl!(agent, [b[1] for b in batch], t2p)
            elseif algo_type == "ICRL"; update_icrl!(agent, batch)
            else; update_ppo!(agent, batch)
            end
        end
        push!(ep_rewards, ep_r)
    end

    last_n = max(1, length(ep_rewards) - 50)
    vr = filter(isfinite, ep_rewards[last_n:end])
    avg_r = isempty(vr) ? -999.0 : mean(vr)
    std_r = (isempty(vr) || length(vr) < 2) ? 0.0 : std(vr)
    return avg_r, std_r, collapses, env.state
end

println("Starting All-Algo Tier 2 Campaign (~900 runs)...")
run_counter = 0
start_time  = now()

for n_agents in N_AGENTS_LIST, seed in SEEDS
    for algo in ["PPO", "SAC", "EGP", "FPRL", "ICRL"]
        global run_counter += 1
        avg_r, std_r, collapses, final_s = run_t2_all_algo(algo, n_agents, seed)
        push!(results_df, (seed, algo, n_agents, avg_r, std_r, collapses, Float64(final_s)))
    end
    
    if run_counter % 60 == 0
        elapsed = now() - start_time
        pct = round(100 * run_counter / total_configs, digits=1)
        println("Progress: $run_counter/$total_configs ($pct%) | Elapsed: $elapsed")
        CSV.write("experiments/results/processed/tier2_all_algo_checkpoint.csv", results_df)
    end
end

CSV.write("experiments/results/processed/tier2_all_algo_full.csv", results_df)

summary_df = combine(groupby(results_df, [:n_agents, :algo]),
    :avg_reward   => mean => :reward_mean,
    :avg_reward   => std  => :reward_std,
    :collapse_count => sum  => :total_collapses,
    :final_resource => mean => :resource_level
)

println("\n--- TIER 2 ALL-ALGORITHM SUMMARY ---")
println(summary_df)
CSV.write("experiments/results/processed/tier2_all_algo_summary.csv", summary_df)

elapsed = now() - start_time
println("\nTier 2 All-Algo Campaign Completed! Duration: $elapsed — Total runs: $(nrow(results_df))")
