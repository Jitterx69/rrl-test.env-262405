module Training

using ..Interfaces
using ..MathUtils
using ..MeasurementUtils
using Printf, Statistics

export ReflexiveTrainer, train!

mutable struct ReflexiveTrainer
    agent
    env
    config
    metrics::Dict{String, Vector{Float32}}
end

function ReflexiveTrainer(agent, env, config)
    # Initialize with advanced metrics tracking
    ReflexiveTrainer(agent, env, config, Dict(
        "consistency" => Float32[], 
        "sensitivity" => Float32[],
        "reward" => Float32[]
    ))
end

function train!(trainer::ReflexiveTrainer)
    c = trainer.config
    epochs = get(c, "epochs", 50)
    steps = get(c, "steps", 100)
    
    for ep in 1:epochs
        batch = []
        s = reset!(trainer.env)
        ep_reward = 0.0f0
        ep_consistency = 0.0f0
        ep_sensitivity = 0.0f0
        
        for t in 1:steps
            # Ensure s is a vector for oracle/policy
            sv = s isa AbstractVector ? Float32.(s) : [Float32(s)]
            
            # Oracle prediction (rho) - multidimensional aware
            rp_vec = trainer.agent.oracle(sv)
            
            # Policy input is concat of state and prediction
            pin = vcat(sv, Float32.(rp_vec))
            rv = trainer.agent.policy(pin)
            mv = rv[1]; svv = rv[2]
            
            # Sample multidimensional action
            a = mv .+ svv .* randn(Float32, length(mv))
            
            # Endogenous transition (step! handles scalar or vector)
            # Use rp_vec[1] for single-alpha environments, or full vector if env expects it.
            # Our current step! methods take a scalar rp or first element.
            rp_pass = length(rp_vec) == 1 ? rp_vec[1] : rp_vec
            sn = step!(trainer.env, a, rp_pass)
            r = reward(trainer.env, sn, a)
            
            # Record advanced metrics
            ep_reward += r
            sn_vec = sn isa AbstractVector ? Float32.(sn) : [Float32(sn)]
            ep_consistency += reflexive_consistency_error(sv, sn_vec)
            
            # Sensitivity calculation
            # Note: feedback_sensitivity currently assumes scalar alpha/rp in most impl.
            # We'll pass the first element to match current measurement_utils.jl impl if needed.
            ep_sensitivity += feedback_sensitivity(trainer.env, sv, a, rp_vec[1])
            
            push!(batch, (sv, Float32.(a), Float32.(rp_vec), sn_vec, Float32(r), 0.0f0))
            s = sn
        end
        
        # Track epoch metrics
        push!(trainer.metrics["reward"], ep_reward / steps)
        push!(trainer.metrics["consistency"], ep_consistency / steps)
        push!(trainer.metrics["sensitivity"], ep_sensitivity / steps)
        
        # Compute returns and update agent
        rtns = compute_returns([b[5] for b in batch])
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(rtns[i]))
        end
        
        # Dispatch updates based on agent type
        # Use full module paths or type checks that are robust to module boundaries
        atype = string(typeof(trainer.agent))
        if occursin("EGPAgent", atype)
            Main.ReflexiveRL.update_egp!(trainer.agent, batch, trainer.env)
        elseif occursin("FPRLAgent", atype)
            Main.ReflexiveRL.update_fprl!(trainer.agent, batch, trainer.env)
        elseif occursin("PPOAgent", atype)
            Main.ReflexiveRL.update_ppo!(trainer.agent, batch)
        elseif occursin("SACAgent", atype)
            Main.ReflexiveRL.update_sac!(trainer.agent, batch)
        elseif occursin("ICRLAgent", atype)
            Main.ReflexiveRL.update_icrl!(trainer.agent, batch)
        end
        
        if ep % 10 == 0
            @printf("Epoch %d | Reward: %.4f | Consistency: %.4f\n", 
                    ep, trainer.metrics["reward"][end], trainer.metrics["consistency"][end])
        end
    end
end

end # module
