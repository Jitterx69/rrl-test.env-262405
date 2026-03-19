module ReflexiveRL

# 1. Base Interfaces & Utils
include("core/interfaces.jl")
using .Interfaces
include("utils/MathUtils.jl")
using .MathUtils
include("utils/measurement_utils.jl")
using .MeasurementUtils

export AbstractReflexiveEnv, AbstractReflexiveAgent
export reset!, step!, reward, observe, compute_returns, compute_advantages
export reflexive_consistency_error, feedback_sensitivity

# 2. Environment Tiers
include("environments/base.jl")
using .Environments
export Tier1Env, Tier2Env, Tier3Env

# 3. Models & Architectures
include("models/architectures.jl")
using .Architectures
export ReflexiveOracle, GaussianPolicy, SpectralOracle

# 4. Algorithms
include("algorithms/egp.jl")
using .EGP
include("algorithms/fprl.jl")
using .FPRL
include("algorithms/ppo.jl")
using .PPO
include("algorithms/sac.jl")
using .SAC
include("algorithms/icrl.jl")
using .ICRL

export EGPAgent, update_egp!
export FPRLAgent, update_fprl!
export PPOAgent, update_ppo!
export SACAgent, update_sac!
export ICRLAgent, update_icrl!

# 5. Training Engine
include("training/Trainer.jl")
using .Training
export ReflexiveTrainer, train!

end # module