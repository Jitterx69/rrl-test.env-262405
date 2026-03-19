module MeasurementUtils

using ..Interfaces
using LinearAlgebra, Statistics

export reflexive_consistency_error, feedback_sensitivity

"""
    reflexive_consistency_error(s_curr, s_next)
Calculates the Mean Squared Error between successive states.
"""
function reflexive_consistency_error(s_curr, s_next)
    return mean((s_curr .- s_next).^2)
end

"""
    feedback_sensitivity(env, state, action, r_pred)
Numerical approximation of the feedback gain dPhi/dr_p.
"""
function feedback_sensitivity(env, state, action, r_pred)
    # Using the explicit Interfaces module path for robustness
    # This prevents any ambiguity with local function names
    eps = 1.0f-4
    
    # Create local copies to avoid mutations during measurement
    e1 = deepcopy(env)
    e0 = deepcopy(env)
    
    s1 = Interfaces.step!(e1, action, r_pred + eps)
    s0 = Interfaces.step!(e0, action, r_pred)
    
    return norm(s1 .- s0) / eps
end

end # module
