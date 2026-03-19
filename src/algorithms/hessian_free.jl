module HessianFree

using LinearAlgebra, ForwardDiff, Zygote
using ..Interfaces

export HessianFreeOptimizer, solve_cg!

"""
    HessianFreeOptimizer(model, loss_f)

An optimizer that uses Conjugate Gradient (CG) on Hessian-Vector Products (HVP)
to perform second-order updates without explicitly storing the Hessian.
"""
struct HessianFreeOptimizer
    model
    loss_f
end

"""
    hvp(f, x, v)

Compute the Hessian-Vector Product H(x)*v using two nested AD passes (Forward-over-Reverse).
H(x)*v = d/deps grad(f)(x + eps*v) | eps=0
"""
function hvp(f, x, v)
    # Gradient of f at x + eps*v
    return ForwardDiff.derivative(eps -> Zygote.gradient(f, x .+ eps .* v)[1], 0.0)
end

"""
    solve_cg!(f, x, b; n_iter=10)

Solve Hx = b for the search direction using Conjugate Gradient.
"""
function solve_cg!(f, x, b, x_init; max_iter=10, tol=1e-5)
    p = copy(b)
    r = b .- hvp(f, x, x_init)
    d = copy(r)
    curr_x = copy(x_init)
    
    for i in 1:max_iter
        Hd = hvp(f, x, d)
        alpha = dot(r, r) / (dot(d, Hd) + 1e-8)
        curr_x .+= alpha .* d
        r_new = r .- alpha .* Hd
        if norm(r_new) < tol; break; end
        beta = dot(r_new, r_new) / (dot(r, r) + 1e-8)
        d .= r_new .+ beta .* d
        r .= r_new
    end
    return curr_x
end

end # module
