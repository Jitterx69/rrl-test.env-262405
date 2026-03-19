using Pkg
Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src"))

using ReflexiveRL
using Flux, Zygote

function test_egp_minimal()
    println("--- Minimal EGP Gradient Test ---")
    agent = EGPAgent(1, 1)
    env = Tier1Env(1.0, 0.05)
    
    # Fake batch: (s, a, rp, sn, r, ret)
    batch = [
        ([0.0f0], [0.1f0], [0.1f0], [0.1f0], 0.0f0, 1.0f0)
    ]
    
    println("Attempting update_egp!...")
    try
        update_egp!(agent, batch, env)
        println("SUCCESS: update_egp! passed.")
    catch e
        println("FAILURE: update_egp! failed: ", e)
        stacktrace()
        rethrow(e)
    end
end

test_egp_minimal()
