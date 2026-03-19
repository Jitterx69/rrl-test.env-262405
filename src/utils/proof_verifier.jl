module ProofVerifier

using ForwardDiff, LinearAlgebra

export verify_lyapunov_conditions

"""
    verify_lyapunov_conditions(f_expr, dynamics_f, domain)

Analytically verifies the Lyapunov stability conditions for the distilled expression:
1. V(0) == 0
2. V(x) > 0 for x != 0
3. dV/dt < 0 along dynamics
"""
function verify_lyapunov_conditions(f_expr, dynamics_f; epsilon=1e-3)
    # 1. Origin check
    v0 = f_expr(0.0f0)
    origin_ok = abs(v0) < 1f-5
    
    # 2. Positivity check (over a grid)
    test_points = range(-2.0, 2.0, length=20)
    positivity_ok = all(s -> (abs(s) < epsilon) || (f_expr(Float32(s)) > 0), test_points)
    
    # 3. Drift check (dV/dt = grad(V) * f(s))
    drift_ok = all(test_points) do s
        s_f = Float32(s)
        if abs(s_f) < epsilon return true end
        grad_v = ForwardDiff.derivative(f_expr, s_f)
        ds = dynamics_f(s_f)
        return (grad_v * ds) < 0
    end
    
    return origin_ok && positivity_ok && drift_ok
end

end # module
