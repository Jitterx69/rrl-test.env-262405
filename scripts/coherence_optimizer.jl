# scripts/coherence_optimizer.jl
# Advanced Synthesis: Global Synchronization Tuning

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, PlotlyJS, Random, Printf

println(">>> Starting Global Coherence Optimization...")

# 1. Setup
N = 1000
alpha_range = range(0.2f0, 1.5f0, length=12)
lambda_range = range(0.01f0, 0.2f0, length=12)

z_coherence = zeros(length(alpha_range), length(lambda_range))
z_tib = zeros(length(alpha_range), length(lambda_range))

# 2. Optimization Loop (Grid Search for population global intelligence)
for (i, alpha) in enumerate(alpha_range), (j, lambda) in enumerate(lambda_range)
    @printf("\r>>> Optimizing: α=%.2f, λ_tib=%.3f ...", alpha, lambda)
    
    env = HolographicReconEnv(N, alpha, 0.1)
    reset!(env)
    
    agent = FNOAgent(1, 1; lr=1f-3, λ_tib=lambda)
    
    coherence_history = Float32[]
    tib_history = Float32[]
    
    # Short train-run
    for t in 1:100
        # Agent collective decisions
        action, r_p = agent(env.state)
        coll_signal = CoherentSpectralLayer(fill(action, N))
        
        # Step environment
        next_s = step!(env, fill(action, N), coll_signal)
        
        # Update agent with TIB
        update_fno!(agent, env.state, action, reward(env, next_s, action), next_s, [env.state])
        
        push!(coherence_history, coll_signal)
        push!(tib_history, estimate_topology_pressure([env.state]))
    end
    
    z_coherence[i, j] = mean(coherence_history[end-20:end])
    z_tib[i, j] = mean(tib_history)
end

println("\n>>> Optimization Complete.")

# 3. Visualization: High-Fidelity 3D Synchronization Manifold
p_opt = PlotlyJS.plot(
    PlotlyJS.surface(x=collect(lambda_range), y=collect(alpha_range), z=z_coherence, 
                     colorscale="Viridis", name="Intelligence Coherence",
                     contours=attr(z=attr(show=true, usecolormap=true, project=attr(z=true)))),
    Layout(title="Global Synchronization Manifold: The Edge of Chaos Frontier",
           scene=attr(xaxis_title="Topological Bottleneck (λ_TIB)",
                      yaxis_title="Coupling Strength (α)",
                      zaxis_title="Spectral Coherence (Intelligence)"),
           margin=attr(l=0, r=0, b=0, t=40))
)

mkpath("experiments/plots/research_manifolds_3d")
PlotlyJS.savefig(p_opt, "experiments/plots/research_manifolds_3d/global_synchronization_3d.html")
println(">>> 3D Manifold Ready: experiments/plots/research_manifolds_3d/global_synchronization_3d.html")

# Also generate a 2D Heatmap version for quick review
p_heat = PlotlyJS.plot(
    PlotlyJS.heatmap(x=collect(lambda_range), y=collect(alpha_range), z=z_coherence, colorscale="Hot"),
    Layout(title="Synchronization Heatmap (α-λ Plane)", xaxis_title="λ_TIB", yaxis_title="α")
)
mkpath("experiments/plots/phase_transitions_heatmaps")
PlotlyJS.savefig(p_heat, "experiments/plots/phase_transitions_heatmaps/synchronization_heatmap.html")

println(">>> Phase 8.1: 3D-HTML Synchronization Manifold generated successfully.")
