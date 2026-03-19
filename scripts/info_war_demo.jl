# scripts/info_war_demo.jl
# Demonstrates Information Warfare where agents manipulate reflexive signals.

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Statistics, PlotlyJS, Random, DataFrames, CSV, Printf

println(">>> Starting Information Warfare Simulation (Reflexive Deception)...")

# 1. Setup
mkpath("experiments/plots")
mkpath("experiments/results/processed")

# Environment: Tier 1 with alpha coupling
alpha = 0.5f0
inner_env = Tier1Env(alpha)
env = CompetitiveEnv(inner_env; alpha_prime=0.2f0)

# Agents: Adversarial vs Standard
agent_a = AdversarialReflexiveAgent(1, 1; lr=1e-3, lambda=0.5f0)
agent_b = EGPAgent(1, 1; lr_o=1e-3, lr_p=1e-3) # Standard agent as the responder

epochs = 100
steps = 50
results = DataFrame(epoch=Int[], reward_a=Float64[], reward_b=Float64[], deception_score=Float64[])

# 2. Training Loop
for ep in 1:epochs
    s = reset!(env)
    ep_r_a = 0.0; ep_r_b = 0.0; ep_dec = 0.0
    
    for t in 1:steps
        # Agent A acts with its adversarial (poisoned) signal
        mv_a, _ = agent_a(s; adversarial=true)
        
        # Agent B responds to the environment (standard reflexive logic)
        st_v_b = s isa AbstractVector ? Float32.(s) : [Float32(s)]
        rp_b = agent_b.oracle(st_v_b)
        mv_b, _ = agent_b.policy(vcat(st_v_b, vec(rp_b)))
        
        # Step environment
        new_s = step!(env, mv_a[1], mv_b[1])
        (r_a, r_b) = reward(env, new_s, mv_a[1], mv_b[1])
        
        # Track deception: Difference between poisoned and natural oracle signal
        rp_natural = agent_a.oracle(s isa Number ? [Float32(s)] : Float32.(s))
        rp_poisoned = rp_natural .+ agent_a.poisoner(vcat(s isa Number ? [Float32(s)] : Float32.(s), vec(rp_natural)))
        ep_dec += mean(abs.(rp_poisoned .- rp_natural))
        
        # Update agents (simplified online update)
        # Note: In real scenarios, these would use a replay buffer
        # For demo, we just track the rewards
        ep_r_a += r_a; ep_r_b += r_b
        s = new_s
    end
    
    push!(results, (ep, ep_r_a/steps, ep_r_b/steps, ep_dec/steps))
    if ep % 20 == 0
        @printf("Epoch %d | R_A: %.3f | R_B: %.3f | Deception: %.3f\n", ep, ep_r_a/steps, ep_r_b/steps, ep_dec/steps)
    end
end

# 3. Visualization: Deception Envelope Manifold
println(">>> Generating Information Warfare Manifold (3D)...")

# Create a grid for the "Deception vs Stability" manifold
# X: Alpha (Coupling), Y: Deception (Signal Noise), Z: System Energy (Loss)
grid_size = 20
alphas = range(0.1, 1.5, length=grid_size)
deceptions = range(0.0, 1.0, length=grid_size)
z_data = zeros(grid_size, grid_size)

for (i, a) in enumerate(alphas), (j, d) in enumerate(deceptions)
    # Theoretical model: High alpha + high deception = Instability (high energy/loss)
    # Low alpha + moderate deception = tactical advantage
    z_data[i, j] = (a-0.5)^2 + (d-0.3)^2 * sign(a-0.8) + 0.1 * a * d
end

p_infowar = PlotlyJS.plot(
    PlotlyJS.surface(x=alphas, y=deceptions, z=z_data, colorscale="Viridis"),
    Layout(
        title="Information Warfare: Deception Envelope Manifold",
        scene=attr(
            xaxis_title="Alpha (Inter-agent Coupling)",
            yaxis_title="Deception Magnitude (Internal)",
            zaxis_title="System Entropy (Energy)"
        )
    )
)

PlotlyJS.savefig(p_infowar, "experiments/plots/info_war_manifold_3d.html")
println(">>> 3D Manifold Ready: experiments/plots/info_war_manifold_3d.html")

# 4. 2D Distribution (SVG)
using Plots
p2d = Plots.plot(results.epoch, [results.reward_a, results.reward_b], 
                label=["Adversarial (Agent A)" "Standard (Agent B)"],
                title="Information Warfare: Competitive Convergence",
                xlabel="Epoch", ylabel="Average Reward",
                lw=2, color=[:red :blue], grid=true)
Plots.savefig(p2d, "experiments/plots/info_war_convergence.svg")
println(">>> 2D Distribution Ready: experiments/plots/info_war_convergence.svg")
