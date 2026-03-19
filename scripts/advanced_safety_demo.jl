# Advanced Stability & Safety (RCBF+) Demo
# Showcasing Neural Lyapunov Manifolds and CBF Safety Filtering
# Focus: Adaptive Safety Boundaries in Non-Linear Dynamics

using Pkg; Pkg.activate(".")
include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Flux, LinearAlgebra
using Statistics
using Plots

println(">>> Starting Advanced Stability & Safety (RCBF+) Demo...")

# 1. Setup Advanced Agent
in_dim = 1
action_dim = 1
agent = NeuralLACAgent(in_dim, action_dim; hidden_dim=32, modes=8, lr=1f-3)

# 2. Define Safety Barrier (Limit state to [-2, 2])
barrier = ControlBarrier(2.0f0)
safety_filter = CBFSafetyFilter(barrier, 0.1f0)

# 3. Simulate and Learn Manifold
function run_demo()
    env = Tier1Env(1.5, 0.05) # high alpha coupling
    s = reset!(env)
    steps = 100
    history_s = Float32[]
    history_v = Float32[]

    println(">>> Learning Neural Lyapunov Manifold...")
    for i in 1:steps
        s_vec = [s]
        a, rp = agent(s_vec)
        
        # Apply Safety Filter (Real-time Guardrail)
        a_safe = safety_filter(s_vec, a, rp, env)
        
        sn = step!(env, a_safe, rp)
        sn_vec = [sn]
        
        # Update Neural Lyapunov and Policy
        update_neural_lac!(agent, s_vec, a_safe, [reward(env, sn, a_safe)], sn_vec)
        
        push!(history_s, s)
        push!(history_v, agent.V_net(s_vec)[1])
        s = sn
    end
    return history_s, history_v
end

history_s, history_v = run_demo()

# 4. Visualizing the Neural Lyapunov Surface (SVG/HTML)
# We sample across the state space [-3, 3]
s_range = collect(-3.0f0:0.1f0:3.0f0)
v_surface = [agent.V_net([x])[1] for x in s_range]

p1 = plot(s_range, v_surface, label="Neural Lyapunov V(s)", title="Learned Stability Manifold", lw=2, color=:green)
vline!([-2.0, 2.0], label="CBF Safety Bounds", ls=:dash, color=:red)
savefig("experiments/plots/neural_lyapunov_manifold.svg")

# 5. Visualizing Trajectory Safety
p2 = plot(history_s, label="State Trajectory", title="CBF-Projected Safe Trajectory", lw=2)
hline!([-2.0, 2.0], label="Safety Bounds", ls=:dash, color=:red)
savefig("experiments/plots/cbf_safe_trajectory.svg")

println(">>> RCBF+ Expansion Demo Complete!")
println("Learned Manifold: experiments/plots/neural_lyapunov_manifold.svg")
println("Safe Trajectory: experiments/plots/cbf_safe_trajectory.svg")
