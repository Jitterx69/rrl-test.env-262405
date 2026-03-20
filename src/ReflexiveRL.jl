module ReflexiveRL

using Printf

# 1. Base Interfaces & Utils
include("core/interfaces.jl")
using .Interfaces
include("utils/MathUtils.jl")
using .MathUtils
include("utils/measurement_utils.jl")
using .MeasurementUtils
include("utils/stability_utils.jl")
using .StabilityUtils
include("utils/accelerator.jl")
using .Accelerator

export AbstractReflexiveEnv, AbstractReflexiveAgent
export reset!, step!, reward, observe, compute_returns, compute_advantages
export reflexive_consistency_error, feedback_sensitivity
export QuadraticLyapunov, LyapunovDrift, ControlBarrier
export NeuralLyapunov, CBFSafetyFilter
export fast_spectral_radius, batch_jvp
include("utils/low_rank.jl")
using .LowRank
include("utils/mixed_precision.jl")
using .MixedPrecision
include("algorithms/hessian_free.jl")
using .HessianFree
include("utils/topology.jl")
using .TopologicalAnalysis
include("utils/symbolic_distiller.jl")
using .SymbolicDistiller
include("utils/proof_verifier.jl")
using .ProofVerifier
include("utils/collective.jl")
using .SuperRadiance
include("environments/population_env.jl")
using .PopulationEnv

export randomized_jacobian_svd, apply_mixed!, HessianFreeOptimizer
export compute_persistence_0d, estimate_topology_pressure
export distill_lyapunov, verify_lyapunov_conditions, evaluate_expression
export CoherentSpectralLayer, compute_population_entropy, MarketPopulationEnv, consensus_proof
export topological_loss, verify_consensus, critical_slowing_index, trigger_shock!

include("environments/recon_env.jl")
using .ReconEnv
export HolographicReconEnv, trigger_shock!

# 2. Environment Tiers
include("environments/base.jl")
using .Environments
export Tier1Env, Tier2Env, Tier3Env

# 3. Models & Architectures
include("models/architectures.jl")
using .Architectures
export ReflexiveOracle, GaussianPolicy, SpectralOracle, GatedSpectralOracle

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
include("algorithms/fno.jl")
using .FNO
include("algorithms/lac.jl")
using .LAC
include("algorithms/neural_lac.jl")
using .NeuralLAC # Added
include("algorithms/adversarial.jl")
include("algorithms/holographic_deception.jl")
include("algorithms/sigint.jl")
include("environments/competitive.jl")

using .EGP, .FPRL, .NeuralLAC, .Adversarial, .HolographicDeception, .SIGINT, .Competitive
export EGPAgent, FPRLAgent, NeuralLACAgent, AdversarialReflexiveAgent
export HolographicDeceiver, SIGINTAgent, CompetitiveEnv, ElectronicWarfareEnv

export EGPAgent, update_egp!
export FPRLAgent, update_fprl!
export PPOAgent, update_ppo!
export SACAgent, update_sac!
export ICRLAgent, update_icrl!
export FNOAgent, update_fno!
export LACAgent, update_lac!
export NeuralLACAgent, update_neural_lac!
export AdversarialReflexiveAgent, update_adversarial!
export CompetitiveEnv

include("algorithms/dynamic_control.jl")
using .DynamicReflexiveControl
export AdaptiveCouplingController, update_coupling!

# 5. Training Engine
include("training/Trainer.jl")
using .Training
export ReflexiveTrainer, train!

end # module