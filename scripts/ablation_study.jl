# Systematic Ablation Study: Isolating Reflexive Components
# Compares: Standard RL, Oracle only, Oracle + Feedback (Reflexive)

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, DataFrames, CSV, Random, Printf

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------

const ALPHAS = [0.0, 0.5, 1.0] # Transition coupling strength
const EPOCHS = 30
const STEPS = 100
const SEEDS = 5

ablation_results = DataFrame(
    condition = String[],
    alpha = Float64[],
    seed = Int[],
    reward = Float64[],
    consistency = Float64[]
)

# ---------------------------------------------------------
# 2. Conditions
# ---------------------------------------------------------

println("Starting Systematic Ablation Study...")

for alpha in ALPHAS
    for seed in 1:SEEDS
        Random.seed!(seed)
        
        # Condition A: Standard RL (No Oracle, No Feedback)
        env_a = Tier1Env(alpha)
        agent_a = PPOAgent(1, 1; lr=1e-3)
        trainer_a = ReflexiveTrainer(agent_a, env_a, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer_a)
        push!(ablation_results, ("Standard RL", alpha, seed, trainer_a.metrics["reward"][end], trainer_a.metrics["consistency"][end]))

        # Condition B: Oracle Signal but No Feedback (Alpha=0 in Env)
        env_b = Tier1Env(0.0) 
        agent_b = EGPAgent(1, 1; lr_o=1e-3, lr_p=1e-3, beta_stab=0.0f0) 
        trainer_b = ReflexiveTrainer(agent_b, env_b, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer_b)
        push!(ablation_results, ("Oracle Only", alpha, seed, trainer_b.metrics["reward"][end], trainer_b.metrics["consistency"][end]))

        # Condition C: Full Reflexive (Oracle + Feedback + Stability Loss)
        env_c = Tier1Env(alpha)
        agent_c = EGPAgent(1, 1; lr_o=1e-3, lr_p=1e-3, beta_stab=0.1f0)
        trainer_c = ReflexiveTrainer(agent_c, env_c, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer_c)
        push!(ablation_results, ("Full Reflexive", alpha, seed, trainer_c.metrics["reward"][end], trainer_c.metrics["consistency"][end]))
        
        @printf("Alpha %.1f | Seed %d | Conditions A-C Complete\n", alpha, seed)
    end
end

mkpath("experiments/results/processed")
CSV.write("experiments/results/processed/ablation_study.csv", ablation_results)

# Summary
summary = combine(groupby(ablation_results, [:condition, :alpha]), :reward => mean => :avg_reward, :consistency => mean => :avg_consistency)
println("\n--- Ablation Summary ---")
println(summary)
CSV.write("experiments/results/processed/ablation_summary.csv", summary)
