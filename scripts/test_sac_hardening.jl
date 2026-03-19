# Test SAC Hardening Verification
using Pkg; Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src"))

using ReflexiveRL
using Statistics, Test

function test_sac_hardening()
    println(">>> Testing SAC Hardening...")
    
    # 1. Test New API Constructor
    agent = SACAgent(1, 1, lr=1e-4)
    @test agent isa AbstractReflexiveAgent
    @test agent.target_entropy == -1.0f0
    println("SUCCESS: New API constructor verified.")
    
    # 2. Test Dummy Update
    env = Tier1Env(1.0)
    batch = []
    # (state, action, rp, next_state, reward, return)
    push!(batch, ([0.1f0], [0.5f0], [0.2f0], [0.6f0], 1.0f0, 1.0f0))
    push!(batch, ([0.2f0], [0.4f0], [0.3f0], [0.7f0], 0.8f0, 0.9f0))
    
    try
        update_sac!(agent, batch)
        println("SUCCESS: update_sac! executed without error.")
    catch e
        @error "FAILURE in update_sac!" exception=e
        rethrow(e)
    end
    
    # 3. Test ReflexiveTrainer Integration
    config = Dict("epochs" => 1, "steps" => 10)
    trainer = ReflexiveTrainer(agent, env, config)
    try
        train!(trainer)
        println("SUCCESS: SAC integrated with ReflexiveTrainer successfully.")
    catch e
        @error "FAILURE in ReflexiveTrainer integration" exception=e
        rethrow(e)
    end
    
    println(">>> SAC HARDENING VERIFIED.")
end

test_sac_hardening()
