# Systematic Ablation Study: Scaling to Tier 1/2/3
# Compares: Standard RL, Oracle only, Oracle + Feedback (Reflexive)

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, DataFrames, CSV, Random, Printf

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------

const TIER = 2 # Change this to 1, 2, or 3 for different experiments
const ALPHAS = [0.0, 0.5, 1.0] 
const EPOCHS = 30
const STEPS = 100
const SEEDS = 5

println(">>> Starting Systematic Ablation Study [Tier $TIER]...")

# Dynamically set dimensions and environment based on TIER
function get_setup(tier, alpha)
    if tier == 1
        return Tier1Env(alpha), 1, 1
    elseif tier == 2
        return Tier2Env(alpha), 1, 1
    elseif tier == 3
        return Tier3Env(alpha), 2, 2
    else
        error("Invalid Tier: $tier")
    end
end

ablation_results = DataFrame(
    condition = String[],
    tier = Int[],
    alpha = Float64[],
    seed = Int[],
    reward = Float64[],
    consistency = Float64[]
)

# ---------------------------------------------------------
# 2. Execution
# ---------------------------------------------------------

for alpha in ALPHAS
    for seed in 1:SEEDS
        Random.seed!(seed)
        env_base, s_dim, a_dim = get_setup(TIER, alpha)
        
        # Condition A: Standard RL (No Oracle, No Feedback Interaction)
        # Note: PPO baseline in our framework still uses a dummy oracle for interface consistency.
        agent_a = PPOAgent(s_dim, a_dim; lr=1e-3)
        trainer_a = ReflexiveTrainer(agent_a, env_base, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer_a)
        push!(ablation_results, ("Standard RL", TIER, alpha, seed, trainer_a.metrics["reward"][end], trainer_a.metrics["consistency"][end]))

        # Condition B: Oracle Only (Minimal Stability Guard)
        agent_b = EGPAgent(s_dim, a_dim; lr_o=1e-3, lr_p=1e-3, beta_stab=0.0f0) 
        trainer_b = ReflexiveTrainer(agent_b, env_base, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer_b)
        push!(ablation_results, ("Oracle Only", TIER, alpha, seed, trainer_b.metrics["reward"][end], trainer_b.metrics["consistency"][end]))

        # Condition C: Full Reflexive (EGP with Stability Regularization)
        agent_c = EGPAgent(s_dim, a_dim; lr_o=1e-3, lr_p=1e-3, beta_stab=0.1f0)
        trainer_c = ReflexiveTrainer(agent_c, env_base, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer_c)
        push!(ablation_results, ("Full Reflexive", TIER, alpha, seed, trainer_c.metrics["reward"][end], trainer_c.metrics["consistency"][end]))
        
        @printf("Tier %d | Alpha %.1f | Seed %d | Complete\n", TIER, alpha, seed)
    end
end

# ---------------------------------------------------------
# 3. Export & Summary
# ---------------------------------------------------------

mkpath("experiments/results/processed")
csv_path = "experiments/results/processed/ablation_study_tier$(TIER).csv"
CSV.write(csv_path, ablation_results)

summary = combine(groupby(ablation_results, [:condition, :alpha]), 
    :reward => mean => :avg_reward, 
    :consistency => mean => :avg_consistency)

println("\n--- Ablation Summary [Tier $TIER] ---")
println(summary)
CSV.write("experiments/results/processed/ablation_summary_tier$(TIER).csv", summary)
