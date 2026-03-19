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
            sv = s isa Vector ? s : [Float32(s)]
            # Oracle prediction (rho)
            rp_vec = trainer.agent.oracle(sv)
            rp = rp_vec[1]
            
            # Policy input is concat of state and prediction
            pin = vcat(sv, [Float32(rp)])
            rv = trainer.agent.policy(pin)
            mv = rv[1]; svv = rv[2]
            a = mv[1] + svv[1] * randn(Float32)
            
            # Endogenous transition
            sn = step!(trainer.env, a, rp)
            r = reward(trainer.env, sn, a)
            
            # Record advanced metrics
            ep_reward += r
            sn_vec = sn isa Vector ? sn : [Float32(sn)]
            ep_consistency += reflexive_consistency_error(sv, sn_vec)
            ep_sensitivity += feedback_sensitivity(trainer.env, sv, a, rp)
            
            push!(batch, (sv, [Float32(a)], [Float32(rp)], sn_vec, Float32(r), 0.0f0))
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
        if trainer.agent isa Main.ReflexiveRL.EGPAgent
            Main.ReflexiveRL.update_egp!(trainer.agent, batch, trainer.env)
        elseif trainer.agent isa Main.ReflexiveRL.FPRLAgent
            Main.ReflexiveRL.update_fprl!(trainer.agent, batch, trainer.env)
        elseif trainer.agent isa Main.ReflexiveRL.PPOAgent
            Main.ReflexiveRL.update_ppo!(trainer.agent, batch)
        end
        
        if ep % 10 == 0
            @printf("Epoch %d | Reward: %.4f | Consistency: %.4f\n", 
                    ep, trainer.metrics["reward"][end], trainer.metrics["consistency"][end])
        end
    end
end

end # module
