# Publication Plot Generation Script (3D Supported)
# Transforms experiment CSVs into research-grade 2D and 3D figures.

using Pkg; Pkg.activate(".")
using Plots, StatsPlots, DataFrames, CSV, Measures, Statistics

# Set publication theme
theme(:vibrant)
default(
    fontfamily="serif", 
    guidefontsize=12, 
    tickfontsize=10, 
    legendfontsize=10, 
    margin=5mm,
    thickness_scaling=1.1,
    grid=true
)

const RESULTS_DIR = "experiments/results/processed"
const PLOT_DIR = "experiments/plots"
mkpath(PLOT_DIR)

println(">>> Generating Publication Plots Library...")

# ---------------------------------------------------------
# 1. Ablation Study Plots (2D)
# ---------------------------------------------------------

function plot_ablation(tier)
    path = joinpath(RESULTS_DIR, tier == 1 ? "ablation_study.csv" : "ablation_study_tier$(tier).csv")
    if isfile(path)
        df = CSV.read(path, DataFrame)
        p = @df df dotplot(:condition, :reward, 
            group=:condition,
            ylabel="Accumulated Reward",
            title="Ablation Criticality: Tier $(tier)",
            legend=:bottomright,
            marker=(:circle, 4, 0.6),
            jitter=0.2
        )
        # Add summary bars
        summary = combine(groupby(df, :condition), :reward => mean => :m, :reward => std => :s)
        @df summary bar!(:condition, :m, yerr=:s, alpha=0.3, color=:grey, label="Mean ± Std")
        
        savefig(p, joinpath(PLOT_DIR, "tier$(tier)_ablation.svg"))
        println("Saved: tier$(tier)_ablation.svg")
    end
end

# ---------------------------------------------------------
# 2. Phase Transition Plots (2D & 3D)
# ---------------------------------------------------------

function plot_phase_transitions_2d(tier)
    path = joinpath(RESULTS_DIR, tier == 1 ? "phase_transitions.csv" : "phase_transitions_tier$(tier).csv")
    if isfile(path)
        df = CSV.read(path, DataFrame)
        p = plot(df.alpha, df.entropy, 
            xlabel="Coupling Strength (α)", ylabel="Trajectory Std",
            title="Stability Bifurcation: Tier $(tier)", lw=2, color=:blue, label="Empirical Variance"
        )
        colors = Dict("Convergent" => :green, "Oscillatory" => :orange, "Divergent" => :red)
        for r in unique(df.regime)
            sub = filter(row -> row.regime == r, df)
            scatter!(sub.alpha, sub.entropy, label=r, color=get(colors, r, :grey), markersize=4)
        end
        savefig(p, joinpath(PLOT_DIR, "tier$(tier)_phase_transition_2d.svg"))
        println("Saved: tier$(tier)_phase_transition_2d.svg")
    end
end

function plot_stability_manifold_3d()
    # Combine Tier 1 and Tier 3 data to show the dimensional stability boundary
    p1 = joinpath(RESULTS_DIR, "phase_transitions.csv")
    p3 = joinpath(RESULTS_DIR, "phase_transitions_tier3.csv")
    
    if isfile(p1) && isfile(p3)
        df1 = CSV.read(p1, DataFrame); df1.dim .= 1
        df3 = CSV.read(p3, DataFrame); df3.dim .= 3 # Dimensional complexity marker
        df = vcat(df1, df3, cols=:intersect)
        
        p = scatter3d(df.alpha, df.dim, df.entropy,
            marker_z=df.entropy,
            xlabel="Alpha (α)", ylabel="Env Complexity (Tier)", zlabel="Variance",
            title="3D Stability Manifold: Dimensional Scaling",
            camera=(45, 30),
            markersize=5,
            color=:turbo,
            colorbar_title="Stability Residual"
        )
        
        # Use .gib extension as requested (PDF-based vector projection for 3D accuracy)
        tmp_path = joinpath(PLOT_DIR, "stability_manifold_3d.pdf")
        savefig(p, tmp_path)
        mv(tmp_path, joinpath(PLOT_DIR, "stability_manifold_3d.gib"), force=true)
        println("Saved: stability_manifold_3d.gib")
    end
end

# ---------------------------------------------------------
# 3. Sensitivity Surface (3D)
# ---------------------------------------------------------

function plot_sensitivity_surface_3d()
    path = joinpath(RESULTS_DIR, "alpha_sensitivity_sweep.csv")
    if isfile(path)
        df = CSV.read(path, DataFrame)
        # Unique algos as Y-axis indices
        algos = unique(df.algo)
        algo_map = Dict(a => i for (i, a) in enumerate(algos))
        df.algo_idx = [algo_map[a] for a in df.algo]
        
        p = scatter3d(df.alpha, df.algo_idx, df.final_reward,
            marker_z=df.final_reward,
            xlabel="Alpha (α)", ylabel="Algorithm Index", zlabel="Final Reward",
            yticks=(1:length(algos), algos),
            title="3D Sensitivity Surface: Algo vs Coupling",
            camera=(30, 45),
            color=:plasma,
            markersize=4,
            label=""
        )
        
        # Use .gib extension as requested (PDF-based vector projection for 3D accuracy)
        tmp_path = joinpath(PLOT_DIR, "sensitivity_surface_3d.pdf")
        savefig(p, tmp_path)
        mv(tmp_path, joinpath(PLOT_DIR, "sensitivity_surface_3d.gib"), force=true)
        println("Saved: sensitivity_surface_3d.gib")
    end
end

# ---------------------------------------------------------
# Execute All
# ---------------------------------------------------------

plot_ablation(1)
plot_ablation(2)
plot_phase_transitions_2d(1)
plot_phase_transitions_2d(3)
plot_stability_manifold_3d()
plot_sensitivity_surface_3d()

println(">>> Visualization Library Generation Complete in $PLOT_DIR")
