# scripts/advanced_info_war_demo.jl
# Advanced Information Warfare: Holographic Deception vs SIGINT

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, PlotlyJS, Random, DataFrames, CSV, Printf

println(">>> Starting Advanced Information Warfare (Spectral Deception vs SIGINT)...")

# 1. Setup
mkpath("experiments/plots")
alpha_base = 0.5f0
# Use a reflexive environment for the core dynamics
tier1 = Tier1Env(alpha_base)
env = ElectronicWarfareEnv(tier1; alpha_base=alpha_base)

# Agents
# HolographicDeceiver: High-tech spectral decoys
agent_deceiver = HolographicDeceiver(1, 1; hidden_dim=32, modes=8, lr=1e-3)
# SIGINTAgent: Adaptive spectral filtering
agent_sigint = SIGINTAgent(1, 1; hidden_dim=32, modes=8, lr=1e-3)

epochs = 50
steps = 40
results = DataFrame(epoch=Int[], reward_dec=Float64[], reward_sig=Float64[], filter_error=Float64[])

# 2. Simulation
for ep in 1:epochs
    s = reset!(env)
    ep_r_dec = 0.0; ep_r_sig = 0.0; ep_f_err = 0.0
    
    for t in 1:steps
        # Deceiver generates poisoned signal
        st_v = [Float32(s)]
        rp_natural = agent_deceiver.oracle(st_v)
        poison = agent_deceiver.spectral_poisoner(vcat(st_v, vec(rp_natural)))
        rp_poisoned = rp_natural .+ poison
        
        # Action for Deceiver
        mv_dec, _ = agent_deceiver.policy(vcat(st_v, vec(rp_poisoned)))
        
        # SIGINT Agent intercepts and filters
        mv_sig, _ = agent_sigint(s, rp_poisoned)
        
        # Jamming: Deceiver tries to jam the coupling if it's losing
        jamming = ep < 25 ? 0.0f0 : 0.4f0
        
        # Step environment
        new_s = step!(env, mv_dec[1], mv_sig[1], jamming)
        r_dec, r_sig = reward(env, new_s, mv_dec[1], mv_sig[1])
        
        # Track filter error (SIGINT's ability to recover natural signal)
        recovered = agent_sigint.denoiser(vcat(st_v, vec(rp_poisoned)))
        ep_f_err += mean(abs.(recovered .- rp_natural))
        
        ep_r_dec += r_dec; ep_r_sig += r_sig
        s = new_s
    end
    
    push!(results, (ep, ep_r_dec/steps, ep_r_sig/steps, ep_f_err/steps))
    if ep % 10 == 0
        @printf("Epoch %d | R_Dec: %.3f | R_Sig: %.3f | FilterErr: %.3f\n", ep, ep_r_dec/steps, ep_r_sig/steps, ep_f_err/steps)
    end
end

# 3. Visualization: Information Supremacy Manifold
println(">>> Generating Information Supremacy Manifold (3D)...")

# Grid: Deception Complexity (Modes) vs SIGINT Filtering (Modes) vs Supremacy (Rel. Reward)
grid_size = 15
dec_modes = range(4, 32, length=grid_size)
sig_modes = range(4, 32, length=grid_size)
z_supremacy = zeros(grid_size, grid_size)

for (i, d) in enumerate(dec_modes), (j, s) in enumerate(sig_modes)
    # Model: If Deception Modes > SIGINT Modes, Deceiver wins (positive supremacy)
    # If SIGINT Modes > Deception Modes, SIGINT wins (negative supremacy)
    z_supremacy[i, j] = (d - s) / 20.0 + 0.1 * randn()
end

p_supremacy = PlotlyJS.plot(
    PlotlyJS.surface(x=dec_modes, y=sig_modes, z=z_supremacy, colorscale="Portland"),
    Layout(
        title="Information Supremacy: Deception vs SIGINT Manifold",
        scene=attr(
            xaxis_title="Deception Spectral Modes",
            yaxis_title="SIGINT Filtering Modes",
            zaxis_title="Supremacy Index (Dec vs Sig)"
        )
    )
)

PlotlyJS.savefig(p_supremacy, "experiments/plots/information_supremacy_3d.html")
println(">>> 3D Manifold Ready: experiments/plots/information_supremacy_3d.html")
