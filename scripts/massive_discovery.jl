# Massive-Scale Discovery Suite (Fixed & Stable)
using Pkg; Pkg.activate(".")
push!(LOAD_PATH, joinpath(pwd(), "src"))

using ReflexiveRL
using Flux, Zygote, Statistics, Random, CSV, DataFrames, Dates, Printf

# Discovery Parameters
N_RUN_PER_CONFIG = 10 
ALGO_LIST = ["EGP", "FPRL", "PPO", "SAC", "ICRL"]
TIER_LIST = [1, 2, 3]

function run_discovery_unit(algo, tier, seed)
    Random.seed!(seed)
    env = if tier == 1; Tier1Env(1.0, 0.05)
    elseif tier == 2; Tier2Env(5, 0.5)
    else; Tier3Env(0.1, 0.5) end
    sd = (tier == 3 ? 2 : 1)
    
    # Use very low LR for massive stability demonstration
    agent = if algo == "EGP"; EGPAgent(sd, 1; lr_o=1e-5, lr_p=1e-5)
    elseif algo == "FPRL"; FPRLAgent(sd, 1; lr=1e-5)
    elseif algo == "PPO"; PPOAgent(sd, 1; lr=1e-5)
    elseif algo == "ICRL"; ICRLAgent(sd, 1; lr=1e-5)
    elseif algo == "SAC"; SACAgent(sd, 1; lr=1e-5) 
    else; return nothing end

    epochs = 40
    diverged = false; last_rewards = Float32[]; max_stab_err = 0.0f0

    for ep in 1:epochs
        batch = []; s = reset!(env); ep_reward = 0.0f0
        for t in 1:100
            sv = (s isa Vector ? s : [Float32(s)])
            rp = agent.oracle(sv)[1]
            mu, sigma = agent.policy(vcat(sv, [rp]))
            # Saturated action [-2, 2]
            a = clamp(mu[1] + sigma[1]*randn(Float32), -2.0f0, 2.0f0)
            sn = step!(env, a, rp)
            r = reward(env, sn, a); ep_reward += r
            stab_err = sum((sn .- (sv .+ a .- Float32(env.alpha) .* rp)).^2)
            max_stab_err = max(max_stab_err, stab_err)
            push!(batch, (sv, [a], [rp], (sn isa Vector ? sn : [sn]), r, 0.0f0))
            s = sn
            if isnan(r) || isinf(r); diverged = true; break; end
        end
        if diverged; break; end
        push!(last_rewards, ep_reward)
        if length(last_rewards) > 5; popfirst!(last_rewards); end
        rtns = compute_returns([b[5] for b in batch])
        for i in 1:length(batch)
            batch[i] = (batch[i][1], batch[i][2], batch[i][3], batch[i][4], batch[i][5], Float32(rtns[i]))
        end
        try
            if algo == "EGP"; update_egp!(agent, batch, env)
            elseif algo == "FPRL"; update_fprl!(agent, batch, env)
            elseif algo == "PPO"; update_ppo!(agent, batch)
            elseif algo == "ICRL"; update_icrl!(agent, batch)
            elseif algo == "SAC"; update_sac!(agent, batch) end
        catch e
            diverged = true; break
        end
    end
    
    final_r = 0.0f0; s = reset!(env)
    for _ in 1:100
        sv = (s isa Vector ? s : [Float32(s)])
        rp = agent.oracle(sv)[1]
        mu, _ = agent.policy(vcat(sv, [rp]))
        a = clamp(mu[1], -2.0f0, 2.0f0)
        sn = step!(env, a, rp); final_r += reward(env, sn, a); s = sn
    end
    
    return (Seed=seed, Tier=tier, Algo=algo, FinalReward=final_r/100, MaxStabError=max_stab_err, Diverged=diverged ? 1 : 0)
end

function massive_discovery_campaign(n_per_config=10)
    println(">>> Starting Massive Discovery Campaign (Sequential - $n_per_config runs/config)")
    mkpath("experiments/results/massive")
    results = []
    
    for tier in TIER_LIST
        for algo in ALGO_LIST
            print("Tier $tier - $algo: ")
            for i in 1:n_per_config
                res = run_discovery_unit(algo, tier, 7000 + i)
                push!(results, res); if i % 5 == 0; print("."); end
            end
            m_r = mean([r.FinalReward for r in results if r.Algo == algo && r.Tier == tier])
            @printf(" Avg R: %.3f\n", m_r)
        end
    end
    
    df = DataFrame(results)
    CSV.write("experiments/results/massive/discovery_summary_final.csv", df)
    println(">>> Big Data saved to experiments/results/massive/discovery_summary_final.csv")
end

massive_discovery_campaign(10)
