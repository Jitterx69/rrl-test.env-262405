# Scaled Tier 2 Campaign: Multi-Agent Resource Allocation
# Exhaustive Parameter Sweep for Research-Grade Publication
# Author: Mohit Ranjan, 2026.
#
# Campaign Design: 1500+ independent simulation runs.
#
# Parameter Matrix:
#   - Seeds:     50 per configuration
#   - N_Agents:  [3, 5, 10]           — Section D.1.2: scalability test
#   - Beta (β):  [0.3, 0.5, 0.8]     — resource regeneration rate sensitivity
#   - Gamma (γ): [0.5, 1.0, 2.0]     — penalty for instability
#   - Algorithms: [PPO, EGP, FPRL]    — 3-way comparison
#
# Total: 50 seeds × 3 N × 3 β × 3 γ × 3 algos = 1215 runs
# (practically 1215 runs, each with 300 episodes × 200 steps)

include("../../src/ReflexiveRL.jl")
using .ReflexiveRL
using CSV, DataFrames, Statistics, Random, Dates

# ---------------------------------------------------------
# 1. Experimental Configuration
# ---------------------------------------------------------

const SEEDS         = 1:50
const N_AGENTS_LIST = [3, 5, 10]
const BETA_LIST     = [0.3, 0.5, 0.8]
const GAMMA_LIST    = [0.5, 1.0, 2.0]
const EPISODES      = 300
const STEPS_PER_EP  = 200
const LR            = 3e-4

# Counters
total_configs = length(SEEDS) * length(N_AGENTS_LIST) * length(BETA_LIST) * length(GAMMA_LIST) * 3
println("Total planned runs: $total_configs")   # 1215

# ---------------------------------------------------------
# 2. Result Collection
# ---------------------------------------------------------

results_df = DataFrame(
    seed              = Int[],
    algo              = String[],
    n_agents          = Int[],
    beta              = Float64[],
    gamma             = Float64[],
    avg_reward        = Float64[],
    collapse_count    = Int[],
    final_resource    = Float64[],
    reward_std        = Float64[]
)

mkpath("experiments/results/processed")

# ---------------------------------------------------------
# 3. Simulation Runner
# ---------------------------------------------------------

function run_t2_config(algo_type::String, n_agents::Int, β::Float64, γ::Float64, seed::Int)
    Random.seed!(seed)
    
    env = Tier2Env(n_agents, β, 0.5, γ)
    oracle = ReflexiveOracle(1)
    policy = GaussianPolicy(1)
    
    agent = if algo_type == "EGP"
        EGPAgent(oracle, policy, LR)
    elseif algo_type == "FPRL"
        FPRLAgent(oracle, policy, LR)
    else
        PPOAgent(policy, LR)
    end
    
    total_rewards = Float64[]
    collapses = 0

    for ep in 1:EPISODES
        s = ReflexiveRL.reset!(env)
        if isnan(s) || isinf(s); s = 0.5; end
        batch = []
        ep_reward = 0.0

        for t in 1:STEPS_PER_EP
            r_pred_raw = algo_type == "PPO" ? 0.0f0 : agent.oracle.model([Float32(s)])[1]
            r_pred = (isnan(r_pred_raw) || isinf(r_pred_raw)) ? 0.0f0 : clamp(r_pred_raw, -2.0f0, 2.0f0)
            
            mu, sigma = agent.policy([Float32(s)])
            mu_val    = isnan(mu[1]) ? 0.0f0 : mu[1]
            sig_val   = (isnan(sigma[1]) || sigma[1] <= 0f0) ? 0.01f0 : sigma[1]
            
            actions = Float32[clamp(mu_val + sig_val * randn(Float32), 0.0f0, Float32(env.a_max)) for _ in 1:n_agents]
            
            s_next = ReflexiveRL.step!(env, actions, r_pred)
            if isnan(s_next) || isinf(s_next); s_next = 0.0; end
            
            rewards = [ReflexiveRL.reward(env, s_next, Float64(actions[i])) for i in 1:n_agents]
            step_r  = isempty(filter(isfinite, rewards)) ? 0.0 : mean(filter(isfinite, rewards))
            ep_reward += step_r
            
            if s_next < 0.05
                collapses += 1
                break
            end
            
            push!(batch, ([Float32(s)], [Float32(mean(actions))], [Float32(r_pred)], [Float32(s_next)], Float32(step_r), 0.0f0))
            s = s_next
        end

        if !isempty(batch)
            rl = [Float64(b[5]) for b in batch]
            rtns = compute_advantages(rl)
            for i in 1:length(batch)
                batch[i] = (batch[i][1],batch[i][2],batch[i][3],batch[i][4],batch[i][5],Float32(rtns[i]))
            end
            if algo_type == "EGP"; update_egp!(agent, batch, env)
            elseif algo_type == "FPRL"
                t2_proxy(S,A,R) = S .- (Float32(n_agents) .* A) .+ Float32(β) .* S .* (1f0 .- S)
                update_fprl!(agent, [b[1] for b in batch], t2_proxy)
            else; update_ppo!(agent, batch)
            end
        end
        push!(total_rewards, ep_reward)
    end
    
    valid_r = filter(isfinite, total_rewards[max(1,end-50):end])
    avg_r   = isempty(valid_r) ? -999.0 : mean(valid_r)
    std_r   = isempty(valid_r) ? 999.0  : std(valid_r)
    return avg_r, std_r, collapses, env.state
end

# ---------------------------------------------------------
# 4. Exhaustive Campaign
# ---------------------------------------------------------

println("Starting Tier 2 Exhaustive Campaign (~1215 runs)...")
run_counter = 0
start_time = now()

for n_agents in N_AGENTS_LIST, β in BETA_LIST, γ in GAMMA_LIST, seed in SEEDS
    for algo in ["PPO", "EGP", "FPRL"]
        global run_counter += 1
        avg_r, std_r, collapses, final_s = run_t2_config(algo, n_agents, β, γ, seed)
        push!(results_df, (seed, algo, n_agents, β, γ, avg_r, collapses, Float64(final_s), std_r))
    end
    
    if run_counter % 90 == 0
        elapsed = now() - start_time
        pct = round(100 * run_counter / total_configs, digits=1)
        println("Progress: $run_counter/$total_configs ($pct%) | Elapsed: $elapsed")
        CSV.write("experiments/results/processed/tier2_exhaustive_checkpoint.csv", results_df)
    end
end

# ---------------------------------------------------------
# 5. Save & Summary
# ---------------------------------------------------------

CSV.write("experiments/results/processed/tier2_exhaustive_full.csv", results_df)

summary_df = combine(groupby(results_df, [:n_agents, :algo]),
    :avg_reward   => mean => :reward_mean,
    :avg_reward   => std  => :reward_std,
    :collapse_count => sum  => :total_collapses,
    :final_resource => mean => :final_level_mean
)

println("\n--- TIER 2 EXHAUSTIVE SUMMARY ---")
println(summary_df)

CSV.write("experiments/results/processed/tier2_exhaustive_summary.csv", summary_df)

elapsed = now() - start_time
println("\nTier 2 Exhaustive Campaign Completed! Duration: $elapsed")
println("Total runs: $(nrow(results_df))")
