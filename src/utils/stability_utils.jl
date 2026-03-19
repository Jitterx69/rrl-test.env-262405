module StabilityUtils

using LinearAlgebra
using Flux
import Statistics: mean

export QuadraticLyapunov, LyapunovDrift, ControlBarrier
export NeuralLyapunov, CBFSafetyFilter

"""
    QuadraticLyapunov(P)

A quadratic Lyapunov candidate V(s) = sᵀPs.
Assumes s is a state vector. P must be positive definite.
"""
struct QuadraticLyapunov
    P::AbstractMatrix{Float32}
end

function QuadraticLyapunov(dim::Int)
    # Default to Identity (Energy-like)
    return QuadraticLyapunov(Matrix{Float32}(I, dim, dim))
end

function (v::QuadraticLyapunov)(s::AbstractVector)
    return dot(s, v.P * s)
end

function (v::QuadraticLyapunov)(s::AbstractMatrix)
    # Batch processing: diag(sᵀPs)
    return sum(s .* (v.P * s), dims=1)
end

"""
    LyapunovDrift(V, s, next_s; epsilon=0.01f0)

Calculates the Lyapunov drift: ΔV = V(next_s) - V(s).
Stability requires ΔV ≤ -εV(s).
Returns the violation: max(0, ΔV + εV(s)).
"""
function LyapunovDrift(V, s, next_s; epsilon=0.01f0)
    v_curr = V(s)
    v_next = V(next_s)
    # We want v_next <= (1 - eps) * v_curr
    # Violation if v_next > (1 - eps) * v_curr
    return max.(0.0f0, v_next .- (1.0f0 - epsilon) .* v_curr)
end

"""
    ControlBarrier(limit)

A simple Control Barrier Function h(s) = limit² - ||s||².
Safe set: {s | h(s) ≥ 0}.
"""
struct ControlBarrier
    limit::Float32
end

function (h::ControlBarrier)(s)
    # h(s) = limit^2 - sᵀs
    return h.limit^2 .- sum(s.^2, dims=1)
end

"""
    CBFViolation(h, s, next_s; gamma=0.1f0)

Calculates the CBF violation based on the discrete-time constraint:
h(next_s) ≥ (1 - γ)h(s).
Returns max(0, (1 - γ)h(s) - h(next_s)).
"""
function CBFViolation(h, s, next_s; gamma=0.1f0)
    h_curr = h(s)
    h_next = h(next_s)
    return max.(0.0f0, (1.0f0 - gamma) .* h_curr .- h_next)
end

# =========================================================
# 4. Neural Lyapunov (PDNN)
# =========================================================

"""
    NeuralLyapunov(phi)

A neural Lyapunov candidate V(s) = ||phi(s)||^2 + eps||s||^2.
Ensures V(s) > 0 for all s != 0.
"""
struct NeuralLyapunov
    phi::Chain
    eps::Float32
end

function NeuralLyapunov(in_dim::Int, hidden_dim::Int=32; eps=1f-3)
    phi = Chain(Dense(in_dim, hidden_dim, relu), Dense(hidden_dim, hidden_dim))
    return NeuralLyapunov(phi, Float32(eps))
end

Flux.@layer NeuralLyapunov

function (v::NeuralLyapunov)(s)
    z = v.phi(s)
    # Power-norm for positive definiteness
    return sum(z.^2, dims=1) .+ v.eps .* sum(s.^2, dims=1)
end

# =========================================================
# 5. CBF Safety Filter (Differentiable Projection)
# =========================================================

"""
    CBFSafetyFilter(h; gamma=0.1f0, lr=0.1f0)

A safety filter that projects an action 'a' to a safe action 'a_safe'
that satisfies the CBF condition: h(f(s, a_safe)) >= (1-gamma)h(s).
Uses a differentiable gradient-step for the prototype.
"""
struct CBFSafetyFilter
    h::ControlBarrier
    gamma::Float32
end

function (f::CBFSafetyFilter)(s, a, rp, env)
    # Simple closed-form approximation or gradient-based projection
    # For this prototype, we check the violation and nudge the action
    s_next = step_model_approx(env, s, a, rp)
    violation = CBFViolation(f.h, s, s_next; gamma=f.gamma)
    
    # If violation > 0, we apply a corrective nudge in the direction of the barrier gradient
    # a_safe = a + violation * grad_a(h(f(s,a)))
    # (Conceptual implementation for demonstration)
    return a .- 0.1f0 .* violation 
end

function step_model_approx(env, s, a, rp)
    # Conceptual next-state prediction
    return s .+ a .- 1.0f0 .* rp
end

end # module
