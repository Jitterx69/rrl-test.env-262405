using YAML
using Random
using Dates
using Flux

# -------------------------
# Load core abstractions FIRST
# -------------------------
include("../src/environments/abstract_env.jl")
include("../src/agents/abstract_agent.jl")

# -------------------------
# Then concrete implementations
# -------------------------
include("../src/environments/tier1_dynamics.jl")
include("../src/models/oracle.jl")
include("../src/models/policy.jl")
include("../src/agents/egp.jl")

# -------------------------
# Utilities
# -------------------------
include("../src/utils/config.jl")
include("../src/utils/logger.jl")
include("../src/utils/seeds.jl")

println("Starting Tier 1 Experiment...")

# -------------------------
# Load config
# -------------------------
config_path = "experiments/configs/tier1/egp_base.yaml"
config = load_config(config_path)

# -------------------------
# Extract parameters
# -------------------------
seeds = config["evaluation"]["seeds"]
episodes = config["training"]["episodes"]
steps = config["training"]["steps_per_episode"]
alpha = config["environment"]["alpha"]

results = []

for (i, seed) in enumerate(seeds)

    println("Running seed: $seed")

    set_seed(seed)

    # -------------------------
    # Setup environment
    # -------------------------
    env = Tier1Env(alpha, 0.0)

    # -------------------------
    # Setup models
    # -------------------------
    oracle_m = Oracle(1, config["model"]["oracle_hidden"])
    policy_m = Policy(config["model"]["policy_hidden"])

    # New Flux/Optimisers.jl style
    opt = Flux.setup(Flux.Adam(config["training"]["learning_rate"]), (oracle_m, policy_m))
    agent = EGPAgent(oracle_m, policy_m, opt)

    # -------------------------
    # Setup logger
    # -------------------------
    run_id = "run_$(lpad(i,3,'0'))"
    log_dir = "experiments/results/raw/tier1/$run_id/"
    logger = init_logger(log_dir)

    # -------------------------
    # Training loop
    # -------------------------
    for ep in 1:episodes

        s = reset!(env)

        total_reward = 0.0
        stability = 0.0
        batch = []

        for t in 1:steps
            r_pred = oracle(agent, s)
            a = policy(agent, r_pred)

            s_next = step!(env, s, a)

            r = reward(env, s, a, s_next)
            
            push!(batch, (s, a, r, s_next))

            total_reward += r
            stability += abs(s - s_next)

            log_trajectory!(logger, t, s, a, r_pred, s_next)

            s = s_next
        end

        # Update agent at the end of episode
        update!(agent, batch)

        log_metrics!(logger, ep, total_reward, stability)
    end

    finalize_logger!(logger)
end

println("Experiment Completed.")