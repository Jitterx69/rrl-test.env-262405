# Phase Transition and Stability Regime Analysis
# Maps the bifurcation points across a high-resolution alpha sweep.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, DataFrames, CSV, Random, Printf

const ALPHA_RANGE = 0.0:0.02:1.6
const EPOCHS = 50
const STEPS = 100
const SEEDS = 3

println("Starting Phase Transition Analysis...")

phase_results = DataFrame(
    alpha = Float64[],
    regime = String[],
    entropy = Float64[],
    divergence_rate = Float64[]
)

for alpha in ALPHA_RANGE
    # Collect trajectory data to detect regimes
    trajectories = []
    
    for seed in 1:SEEDS
        Random.seed!(seed)
        env = Tier1Env(alpha)
        agent = EGPAgent(1, 1; lr_o=1e-3, lr_p=1e-3)
        trainer = ReflexiveTrainer(agent, env, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer)
        
        # Sample final trajectory
        s = reset!(env)
        traj = []
        for t in 1:STEPS
            rp = agent.oracle([s])[1]
            a = agent.policy(vcat([s], [rp]))[1][1]
            s = step!(env, a, rp)
            push!(traj, s)
        end
        push!(trajectories, traj)
    end
    
    # Analyze trajectories for regime detection
    avg_traj_std = mean([std(t[end-20:end]) for t in trajectories])
    is_divergent = any([any(abs.(t) .>= 9.9) for t in trajectories])
    
    regime = if is_divergent
        "Divergent"
    elseif avg_traj_std > 0.5
        "Oscillatory"
    else
        "Convergent"
    end
    
    push!(phase_results, (alpha, regime, avg_traj_std, is_divergent ? 1.0 : 0.0))
    @printf("Alpha %.2f | Regime: %s\n", alpha, regime)
end

mkpath("experiments/results/processed")
CSV.write("experiments/results/processed/phase_transitions.csv", phase_results)
println("Analysis complete. Results exported to experiments/results/processed/phase_transitions.csv")
