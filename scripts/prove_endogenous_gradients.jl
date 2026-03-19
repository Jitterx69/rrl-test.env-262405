# Proof of Endogenous Gradients: Bias Correction Analysis
# This script compares Standard PG (non-differentiable transition) 
# against Endogenous PG (differentiable through Zygote).

include("../src/ReflexiveRL.jl")
using .ReflexiveRL
using Flux, Zygote, Statistics, Random, Printf

# ---------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------

const ALPHA = 1.0f0      # Strong coupling to highlight bias
const TARGET = 5.0f0     # Convergence target
const STEPS = 10         # Short horizon for clear gradient analysis
const EPOCHS = 50
const LR = 1e-3

# ---------------------------------------------------------
# 2. Gradient Comparison Logic
# ---------------------------------------------------------

function calculate_gradient(env, oracle, policy, endogenous::Bool)
    s = 0.0f0 
    batch_loss = 0.0f0
    
    gs = Zygote.gradient(oracle, policy) do o, p
        loss = 0.0f0
        curr_s = s
        for t in 1:STEPS
            rp = o([curr_s])[1]
            mv, sv = p(vcat([curr_s], [rp]))
            a = mv[1]
            
            # Transition
            if endogenous
                # Differentiable pathway
                next_s = curr_s + a - ALPHA * rp
            else
                # Block gradient through transition (Standard PG)
                next_s = Zygote.ignore() do
                    curr_s + a - ALPHA * rp
                end
            end
            
            # Distance-to-target loss
            loss += (next_s - TARGET)^2
            curr_s = next_s
        end
        return loss / STEPS
    end
    return gs
end

# ---------------------------------------------------------
# 3. Execution & Comparison
# ---------------------------------------------------------

println("Starting Gradient Bias Analysis (Endogenous vs. Standard)...")

# Initialize identical models
o_std = ReflexiveOracle(1); p_std = GaussianPolicy(2)
o_end = deepcopy(o_std); p_end = deepcopy(p_std)

opt_std = Flux.setup(Flux.Adam(LR), (o_std, p_std))
opt_end = Flux.setup(Flux.Adam(LR), (o_end, p_end))

history_std = []; history_end = []

for ep in 1:EPOCHS
    # Standard Update
    gs_std = calculate_gradient(nothing, o_std, p_std, false)
    Flux.update!(opt_std, (o_std, p_std), gs_std)
    
    # Endogenous Update
    gs_end = calculate_gradient(nothing, o_end, p_end, true)
    Flux.update!(opt_end, (o_end, p_end), gs_end)
    
    # Track performance (Target RMSE)
    s_std = 0.0f0; s_end = 0.0f0
    for t in 1:STEPS
        a_s = o_std([s_std])[1]; s_std = s_std + p_std(vcat([s_std], [a_s]))[1][1] - ALPHA * a_s
        a_e = o_end([s_end])[1]; s_end = s_end + p_end(vcat([s_end], [a_e]))[1][1] - ALPHA * a_e
    end
    
    push!(history_std, abs(s_std - TARGET))
    push!(history_end, abs(s_end - TARGET))
    
    if ep % 10 == 0
        @printf("Epoch %d | Bias (Std): %.4f | Bias (End): %.4f\n", 
                ep, history_std[end], history_end[end])
    end
end

println("\nAnalysis Complete.")
if history_end[end] < history_std[end]
    println("SUCCESS: Endogenous PG achieved lower convergence bias than Standard PG.")
else
    println("WARNING: Target bias did not significantly improve. Check coupling strength (ALPHA).")
end
