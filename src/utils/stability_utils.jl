module StabilityUtils

using LinearAlgebra
using Flux

export QuadraticLyapunov, LyapunovDrift, ControlBarrier

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

end # module
