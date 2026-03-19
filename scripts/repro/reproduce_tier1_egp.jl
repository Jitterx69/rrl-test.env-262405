# Tier 1 EGP Reproduction (High-End Implementation)
# Validates Section 9.2 of Manuscript Draft I
# Author: Mohit Ranjan, 2026.

using Pkg
Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src")) # Ensure local source is visible

using ReflexiveRL
using Statistics, Printf

function run_tier1_egp_benchmark()
    println("=== Tier 1 EGP High-End Benchmark ===")
    
    # 1. Configuration
    config = Dict(
        "epochs" => 100,
        "steps" => 200,
        "lr_oracle" => 1e-4,
        "lr_policy" => 3e-4,
        "alpha" => 1.0 # High reflexive gain (stress test)
    )
    
    # 2. Environment & Agent setup
    env = Tier1Env(config["alpha"], 0.05)
    agent = EGPAgent(1, 1, lr_o=config["lr_oracle"], lr_p=config["lr_policy"])
    
    # 3. Trainer initialization
    trainer = ReflexiveTrainer(agent, env, config)
    
    # 4. Execution
    println(">>> Commencing Training...")
    train!(trainer)
    
    println("\n=== Final Performance & Stability Report ===")
    # Final evaluation (no policy updates)
    s = reset!(env)
    test_rewards = []
    stab_errors = []
    
    for t in 1:1000
        st_v = [Float32(s)]
        r_pred = agent.oracle(st_v)[1]
        p_in = vcat(st_v, [Float32(r_pred)])
        m_v, s_v = agent.policy(p_in)
        a = m_v[1] # Use mean action for final eval
        
        s_next = step!(env, a, r_pred)
        r = reward(env, s_next, a)
        
        # Phi(s) = s + a - alpha * r_pred
        s_next_pred = s + a - env.alpha * r_pred
        push!(stab_errors, abs(s_next - s_next_pred))
        
        push!(test_rewards, r)
        s = s_next
    end
    
    @printf("Mean Test Reward: %.3f\n", mean(test_rewards))
    @printf("Final E_stab (Consistency Error): %.4f\n", mean(stab_errors))
    
    if mean(stab_errors) < 0.1
        println(">>> VERIFIED: EGP successfully minimized reflexive stability error.")
    else
        println(">>> WARNING: EGP stability error remains high. Check reflexive gain G.")
    end
end

run_tier1_egp_benchmark()
