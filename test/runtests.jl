push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using ReflexiveRL
using ReflexiveRL.Interfaces # Explicitly pull in Interfaces
using Test, Statistics, Random, Flux, Zygote

@testset "ReflexiveRL.jl Formal Suite" begin

    @testset "Math Utilities" begin
        # 1. compute_returns
        rewards = [1.0, 1.0, 1.0]
        gamma = 0.9
        returns = compute_returns(rewards, gamma)
        @test returns[3] ≈ 1.0
        
        # 2. MeasurementUtils
        s1 = [1.0f0, 2.0f0]; s2 = [1.1f0, 1.9f0]
        @test reflexive_consistency_error(s1, s2) ≈ 0.01f0
        
        env = Tier1Env(1.0, 0.0)
        # fs should be approx alpha = 1.0
        # Use Float32 for all inputs to match framework standards
        fs = feedback_sensitivity(env, [0.0f0], [1.0f0], 0.0f0)
        @test fs ≈ 1.0 rtol=1e-2
    end

    @testset "Environment Tiers" begin
        # Tier 1
        env1 = Tier1Env(1.0, 0.0)
        s0 = reset!(env1)
        @test s0 == 0.0f0
        # Use Vector action since Trainer/Agents now push Vectors
        s1 = step!(env1, [0.5f0], 0.0f0)
        @test s1 == 0.5f0
        @test reward(env1, s1, [0.5f0]) ≈ -0.5^2 - 0.1*0.5^2
        
        # Tier 2
        env2 = Tier2Env(1.0, 0.0)
        reset!(env2)
        s1_2 = step!(env2, [0.0f0], 0.0f0)
        @test s1_2 ≈ 0.0f0
        
        # Tier 3
        env3 = Tier3Env(0.1, 0.0)
        s0_3 = reset!(env3)
        @test length(s0_3) == 2
        s1_3 = step!(env3, [1.0f0, 0.5f0], [0.0f0, 0.0f0])
        @test s1_3 ≈ [1.0f0, 0.5f0]
    end

    @testset "Architectures" begin
        orc = ReflexiveOracle(2, 1)
        @test length(orc([1.0f0, 1.0f0])) == 1
        
        pol = GaussianPolicy(2, 1)
        mv, sv = pol([1.0f0, 1.0f0])
        @test length(mv) == 1
        @test all(sv .> 0)
        
        gs = Zygote.gradient(orc) do o
            sum(o([1.0f0, 1.0f0]))
        end
        @test gs[1] !== nothing
    end

    @testset "Agent Sanity" begin
        batch = [([1.0f0], [0.5f0], [0.1f0], [1.1f0], 1.0f0, 1.0f0)]
        env = Tier1Env(0.1, 0.0)
        
        @test (update_egp!(EGPAgent(1, 1), batch, env); true) 
        @test (update_ppo!(PPOAgent(1, 1), batch); true)
        @test (update_fprl!(FPRLAgent(1, 1), batch, env); true)
        @test (update_sac!(SACAgent(1, 1), batch); true)
        @test (update_icrl!(ICRLAgent(1, 1), batch); true)
    end

    @testset "Training Engine Integration" begin
        env = Tier1Env(0.1, 0.0)
        agent = EGPAgent(1, 1)
        trainer = ReflexiveTrainer(agent, env, Dict("epochs" => 1, "steps" => 2))
        @test (train!(trainer); true)
        @test length(trainer.metrics["reward"]) == 1
    end

end
