# scripts/resilience_shock_demo.jl
# Phase 8: Resilience Stress Testing & Recovery Visualization

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, PlotlyJS, Random, Printf

println(">>> Starting Resilience Shock Stress Test...")

# 1. Setup
N = 1000
env = HolographicReconEnv(N, 0.9f0, 0.1f0)
reset!(env)

history_state = Float32[]
history_coherence = Float32[]
history_time = Int[]

# 2. Simulation with Shock
println(">>> Running simulation with informational shock at T=100...")

for t in 1:250
    # Simulate agents
    signals = randn(Float32, N) .* 0.1f0 .+ env.state
    coll_signal = CoherentSpectralLayer(signals)
    
    # Inject Shock
    if t == 100
        trigger_shock!(env; magnitude=2.5f0)
    end
    
    # Step Environment
    s = step!(env, signals, coll_signal)
    
    push!(history_state, s)
    push!(history_coherence, coll_signal)
    push!(history_time, t)
end

# 3. Visualization: Recovery Trajectory (3D Path)
# We plot State vs Coherence vs Time to show the 'Snap Back' to stability.
p_shock = PlotlyJS.plot(
    PlotlyJS.scatter3d(x=history_time, y=history_state, z=history_coherence,
                       mode="lines+markers", 
                       marker=attr(size=3, color=history_time, colorscale="Plotly3"),
                       line=attr(width=4, color="blue")),
    Layout(title="Resilience Recovery Trajectory: Response to Informational Shock",
           scene=attr(xaxis_title="Time (T)",
                      yaxis_title="Env State",
                      zaxis_title="Intelligence Coherence"),
           margin=attr(l=0, r=0, b=0, t=40))
)

mkpath("experiments/plots/formal_stability_safety")
PlotlyJS.savefig(p_shock, "experiments/plots/formal_stability_safety/recovery_trajectory_plot.html")
println(">>> Recovery Plot Ready: experiments/plots/formal_stability_safety/recovery_trajectory_plot.html")

println(">>> Phase 8.4: Resilience Shock Stress Test completed.")
