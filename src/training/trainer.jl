module Training

using ..Interfaces
using ..MathUtils
using Printf, Statistics

export ReflexiveTrainer, train!

mutable struct ReflexiveTrainer
    agent
    env
    config
end

function train!(trainer::ReflexiveTrainer)
    c = trainer.config
    epochs = get(c, "epochs", 50)
    steps = get(c, "steps", 100)
    
    for ep in 1:epochs
        batch = []
        s = reset!(trainer.env)
        for t in 1:steps
            sv = s isa Vector ? s : [Float32(s)]
            rp = trainer.agent.oracle(sv)[1]
            pin = vcat(sv, [Float32(rp)])
            # Renaming for consistency
            rv = trainer.agent.policy(pin)
            mv = rv[1]; svv = rv[2]
            a = mv[1] + svv[1] * randn(Float32)
            
            sn = step!(trainer.env, a, rp)
            r = reward(trainer.env, sn, a)
            
            snv = sn isa Vector ? sn : [Float32(sn)]
            push!(batch, (sv, [Float32(a)], [Float32(rp)], snv, Float32(r), 0.0f0))
            s = sn
        end
        
        rtns = compute_returns([b[5] for b in batch])
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(rtns[i]))
        end
        
        # Dispatch updates
        if trainer.agent isa Main.ReflexiveRL.EGPAgent
            Main.ReflexiveRL.update_egp!(trainer.agent, batch, trainer.env)
        elseif trainer.agent isa Main.ReflexiveRL.FPRLAgent
            Main.ReflexiveRL.update_fprl!(trainer.agent, batch, trainer.env)
        elseif trainer.agent isa Main.ReflexiveRL.PPOAgent
            Main.ReflexiveRL.update_ppo!(trainer.agent, batch)
        end
    end
end

end # module
