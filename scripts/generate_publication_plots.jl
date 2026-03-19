# Publication Plot Generation Script (Comprehensive Research Suite)
# GR for 2D Vector (SVG) | Plotly for 3D Interactive (HTML)

using Pkg; Pkg.activate(".")
using Plots, StatsPlots, DataFrames, CSV, Measures, Statistics

const RESULTS_DIR = "experiments/results/processed"
const PLOT_DIR = "experiments/plots"
mkpath(PLOT_DIR)

println(">>> Generating Comprehensive Publication Plots Library...")

# ---------------------------------------------------------
# 1. Ablation & Single-Tier 2D Plots (SVG via GR)
# ---------------------------------------------------------

function plot_ablation(tier)
    gr()
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
        summary = combine(groupby(df, :condition), :reward => mean => :m, :reward => std => :s)
        @df summary bar!(:condition, :m, yerr=:s, alpha=0.3, color=:grey, label="Mean ± Std")
        savefig(p, joinpath(PLOT_DIR, "tier$(tier)_ablation.svg"))
        println("Saved: tier$(tier)_ablation.svg")
    end
end

# ---------------------------------------------------------
# 2. Benchmarking Plots (2D SVG)
# ---------------------------------------------------------

function plot_algorithm_benchmarks()
    gr()
    for tier in [1, 2, 3]
        path = joinpath(RESULTS_DIR, "tier$(tier)_all_algo_summary.csv")
        if isfile(path)
            df = CSV.read(path, DataFrame)
            # Plot at max alpha or average
            p = @df df groupedbar(:algo, :reward_mean, 
                group=:algo,
                ylabel="Mean Reward",
                title="Performance Benchmark: Tier $(tier)",
                yerr=:reward_std,
                legend=false,
                color=:auto
            )
            savefig(p, joinpath(PLOT_DIR, "tier$(tier)_algo_benchmark.svg"))
            println("Saved: tier$(tier)_algo_benchmark.svg")
        end
    end
end

# ---------------------------------------------------------
# 3. 3D Manifolds & Interactive Surfaces (HTML via Plotly)
# ---------------------------------------------------------

function plot_stability_manifold_3d()
    plotly()
    p1 = joinpath(RESULTS_DIR, "phase_transitions.csv")
    p3 = joinpath(RESULTS_DIR, "phase_transitions_tier3.csv")
    
    if isfile(p1) && isfile(p3)
        df1 = CSV.read(p1, DataFrame); df1.dim .= 1
        df3 = CSV.read(p3, DataFrame); df3.dim .= 3 
        df = vcat(df1, df3, cols=:intersect)
        
        p = scatter3d(df.alpha, df.dim, df.entropy,
            marker_z=df.entropy,
            xlabel="Alpha (α)", ylabel="Env Complexity (Tier)", zlabel="Variance",
            title="3D Stability Manifold (Interactive)",
            color=:turbo
        )
        savefig(p, joinpath(PLOT_DIR, "stability_manifold_3d.html"))
        println("Saved: stability_manifold_3d.html")
    end
end

function plot_exhaustive_surfaces_3d()
    plotly()
    for tier in [2, 3]
        path = joinpath(RESULTS_DIR, "tier$(tier)_exhaustive_full.csv")
        if isfile(path)
            df = CSV.read(path, DataFrame)
            if "beta" in names(df) && "gamma" in names(df)
                p = scatter3d(df.beta, df.gamma, df.avg_reward,
                    marker_z=df.avg_reward,
                    xlabel="Beta (β)", ylabel="Gamma (γ)", zlabel="Reward",
                    title="Exhaustive Param Space: Tier $(tier)",
                    color=:magma
                )
                savefig(p, joinpath(PLOT_DIR, "tier$(tier)_exhaustive_surface.html"))
                println("Saved: tier$(tier)_exhaustive_surface.html")
            end
        end
    end
end

function plot_sensitivity_surface_3d()
    plotly()
    path = joinpath(RESULTS_DIR, "alpha_sensitivity_sweep.csv")
    if isfile(path)
        df = CSV.read(path, DataFrame)
        algos = unique(df.algo)
        algo_map = Dict(a => i for (i, a) in enumerate(algos))
        df.algo_idx = [algo_map[a] for a in df.algo]
        
        p = scatter3d(df.alpha, df.algo_idx, df.final_reward,
            marker_z=df.final_reward,
            xlabel="Alpha (α)", ylabel="Algorithm Index", zlabel="Final Reward",
            yticks=(1:length(algos), algos),
            title="Algorithm Sensitivity Landscape",
            color=:plasma,
            label=""
        )
        savefig(p, joinpath(PLOT_DIR, "sensitivity_surface_3d.html"))
        println("Saved: sensitivity_surface_3d.html")
    end
end

# ---------------------------------------------------------
# Execute All
# ---------------------------------------------------------

# 2D Plots
plot_ablation(1)
plot_ablation(2)
plot_algorithm_benchmarks()

# 3D Plots
plot_stability_manifold_3d()
plot_exhaustive_surfaces_3d()
plot_sensitivity_surface_3d()

println(">>> Comprehensive Visualization Library Complete (SVG/HTML) in $PLOT_DIR")
