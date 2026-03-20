module DynamicReflexiveControl

using Statistics, LinearAlgebra
using ..Interfaces

export AdaptiveCouplingController, update_coupling!

"""
    AdaptiveCouplingController(target_csd, kp, ki)

A PID-based meta-controller that adjusts the environment's coupling strength (alpha)
to maintain a target 'Critical Slowing Down' (CSD) index.
Targeting the phase transition point (edge of chaos) maximizes information coherence.
"""
mutable struct AdaptiveCouplingController
    target_csd::Float32
    kp::Float32
    ki::Float32
    integral_error::Float32
    last_alpha::Float32
end

function AdaptiveCouplingController(target_csd=0.7f0; kp=0.1f0, ki=0.01f0)
    return AdaptiveCouplingController(target_csd, kp, ki, 0.0f0, 0.5f0)
end

"""
    update_coupling!(controller, current_csd, env)

Adjusts the environment's alpha based on the measured CSD.
If CSD > target: System is too stable/slow -> Increase coupling (alpha).
If CSD < target: System is becoming chaotic -> Decrease coupling (alpha).
"""
function update_coupling!(controller::AdaptiveCouplingController, current_csd, env::AbstractReflexiveEnv)
    error = controller.target_csd - current_csd
    controller.integral_error += error
    
    # PID update (proportional-integral)
    adjustment = controller.kp * error + controller.ki * controller.integral_error
    
    # New alpha
    new_alpha = clamp(controller.last_alpha - adjustment, 0.1f0, 2.0f0)
    
    # Apply to environment if it supports alpha modification
    if hasproperty(env, :alpha)
        env.alpha = new_alpha
    end
    
    controller.last_alpha = new_alpha
    return new_alpha
end

end # module
