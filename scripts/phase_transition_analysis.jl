# Phase Transition and Stability Regime Analysis: Scaling to Tier 1/2/3
# Maps the bifurcation points across a high-resolution alpha sweep.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, DataFrames, CSV, Random, Printf, LinearAlgebra

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------

const TIER = 3
const ALPHA_RANGE = 0.0:0.04:1.6
const EPOCHS = 50
const STEPS = 100
const SEEDS = 3

println(">>> Starting Phase Transition Analysis [Tier $TIER]...")

# Dynamically set dimensions and environment based on TIER
function get_setup(tier, alpha)
    if tier == 1
        return Tier1Env(alpha), 1, 1, 0.5
    elseif tier == 2
        return Tier2Env(alpha), 1, 1, 0.8 
    elseif tier == 3
        return Tier3Env(alpha, 0.1), 2, 2, 0.3 
    else
        error("Invalid Tier: $tier")
    end
end

phase_results = DataFrame(
    tier = Int[],
    alpha = Float64[],
    regime = String[],
    entropy = Float64[],
    divergence_rate = Float64[]
)

# ---------------------------------------------------------
# 2. Execution
# ---------------------------------------------------------

for alpha in ALPHA_RANGE
    trajectories = []
    env_sample, s_dim, a_dim, osc_threshold = get_setup(TIER, alpha)
    
    for seed in 1:SEEDS
        Random.seed!(seed)
        env, _, _, _ = get_setup(TIER, alpha)
        agent = EGPAgent(s_dim, a_dim; lr_o=3e-4, lr_p=3e-4) # Lowered LR for Tier 3 stability
        trainer = ReflexiveTrainer(agent, env, Dict("epochs"=>EPOCHS, "steps"=>STEPS))
        train!(trainer)
        
        # Sample final trajectory
        s = reset!(env)
        traj = []
        for t in 1:STEPS
            st_v = s isa AbstractVector ? Float32.(s) : [Float32(s)]
            rp_vec = agent.oracle(st_v)
            input_p = vcat(st_v, Float32.(rp_vec))
            rv = agent.policy(input_p)
            mv = rv[1]
            # Use mean action for stability profiling
            s = step!(env, mv, rp_vec)
            push!(traj, copy(s))
        end
        push!(trajectories, traj)
    end
    
    # Analyze trajectories for regime detection
    function get_std(t)
        if t[1] isa Number
            return std(t[end-20:end])
        else
            # For 2D, take the mean of coordinate-wise std
            return (std([x[1] for x in t[end-20:end]]) + std([x[2] for x in t[end-20:end]])) / 2
        end
    end
    
    avg_traj_std = mean([get_std(t) for t in trajectories])
    
    # Divergence detection
    is_divergent(t) = any(x -> any(abs.(x) .>= 9.9), t)
    div_count = count(is_divergent, trajectories)
    
    regime = if div_count > SEEDS/2
        "Divergent"
    elseif avg_traj_std > osc_threshold
        "Oscillatory"
    else
        "Convergent"
    end
    
    push!(phase_results, (TIER, alpha, regime, avg_traj_std, div_count/SEEDS))
    @printf("Tier %d | Alpha %.2f | Regime: %s (Std: %.3f)\n", TIER, alpha, regime, avg_traj_std)
end

# ---------------------------------------------------------
# 3. Export
# ---------------------------------------------------------

mkpath("experiments/results/processed")
CSV.write("experiments/results/processed/phase_transitions_tier$(TIER).csv", phase_results)
println("Analysis complete. Results exported to experiments/results/processed/phase_transitions_tier$(TIER).csv")
