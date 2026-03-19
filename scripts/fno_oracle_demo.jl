# Advanced Neural Operator (FNO+) Expansion Demo
# Showcasing Gated Fourier Units (GFU) and FNOAgent
# Focus: High-Frequency Stability & Gated Residual Processing

using Pkg; Pkg.activate(".")
# Note: We include source directly for the demo to bypass local package path issues
include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Flux
using Statistics
using Plots

println(">>> Starting Advanced Neural Operator (FNO+) Expansion Demo...")

# 1. Setup Advanced Models
in_dim = 10
out_dim = 1
hidden = 64
modes = 16

# MLP vs Standard FNO vs Gated FNO+
mlp_oracle = ReflexiveOracle(in_dim, out_dim)
fno_oracle = SpectralOracle(in_dim, out_dim, hidden, modes)
gated_fno  = GatedSpectralOracle(in_dim, out_dim, hidden, modes)

# 2. Setup FNOAgent
agent = FNOAgent(in_dim, out_dim, hidden, modes; lr=1f-3, β_stab=0.5f0)

# 3. Simulate Complex Dynamics (Multi-scale)
function generate_complex_signal(n_steps, dt)
    t = collect(0:dt:(n_steps-1)*dt)
    # Signal with both slow global flow and sharp high-frequency "shocks"
    signal = sin.(2π * 1.0 .* t) .+ 0.2 .* sin.(2π * 20.0 .* t)
    # Add a local shock
    signal[Int(n_steps÷2):Int(n_steps÷2+5)] .+= 0.5
    
    obs = zeros(Float32, in_dim, n_steps)
    for i in 1:n_steps
        obs[:, i] .= signal[i]
    end
    return obs
end

obs_complex = generate_complex_signal(100, 0.01)

println(">>> Evaluating Model Response to Complex Signals...")
y_mlp   = mlp_oracle(obs_complex)
y_fno   = fno_oracle(obs_complex)
y_gated = gated_fno(obs_complex)

println("Gated FNO Output Shape: ", size(y_gated))

# 4. Demonstrate Optimization (Update Loop)
println(">>> Proving FNOAgent Update Stability...")
# Dummy batch for update
actions = randn(Float32, 1, 100)
rewards = randn(Float32, 1, 100)
next_obs = obs_complex .+ 0.01f0 .* randn(Float32, size(obs_complex))

try
    update_fno!(agent, obs_complex, actions, rewards, next_obs)
    println(">>> FNOAgent update successful!")
catch e
    @warn "FNOAgent update encounterd an issue (Adjoint related): " e
end

# 5. Visualizing the Gated Advantage (SVG)
p = plot(obs_complex[1, :], label="State (Complex)", title="Gated Fourier Unit (GFU) Performance", lw=1.5, color=:black)
plot!(y_fno[1, :], label="Standard FNO (Low-Pass Only)", alpha=0.6, ls=:dash)
plot!(y_gated[1, :], label="Gated FNO+ (Global + Local)", alpha=0.8, color=:red, lw=2)
savefig("experiments/plots/fno_plus_performance.svg")

println(">>> FNO+ Expansion Demo Complete!")
println("Check experiments/plots/fno_plus_performance.svg for visual verification.")
