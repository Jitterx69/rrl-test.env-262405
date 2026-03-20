module SymbolicDistiller

using LinearAlgebra, Statistics, Printf

export distill_lyapunov, evaluate_expression

"""
    distill_lyapunov(oracle, states)

Fits the neural oracle's outputs to a set of symbolic Lyapunov templates.
Uses a multi-template least-squares approach to find the best analytical fit.
"""
function distill_lyapunov(oracle, states)
    # 1. Samples from neural oracle
    # states is expected to be [dim, N] or [N]
    outputs = vec(oracle(states))
    s_vec = vec(states)
    
    # 2. Define Templates
    # We fit: V(s) = c1*T1(s) + c2*T2(s) + ...
    # Template library
    templates = [
        s -> s.^2,                          # Quadratic
        s -> exp.(s) .+ exp.(-s) .- 2.0f0,  # Exponential Barrier
        s -> log.(cosh.(s)),                # Tanh Energy
        s -> abs.(s)                        # L1-like (non-smooth)
    ]
    template_names = ["s^2", "(exp(s)+exp(-s)-2)", "log(cosh(s))", "|s|"]
    
    best_residual = Inf
    best_coeffs = nothing
    best_template_idx = 0
    
    # Simple search: single template or linear combination
    for i in 1:length(templates)
        # Construct Design Matrix A for single template
        A = templates[i](s_vec)
        # Solve A * c = outputs => c = A \ outputs
        c = A \ outputs
        
        residual = norm(A * c .- outputs)
        if residual < best_residual
            best_residual = residual
            best_coeffs = c
            best_template_idx = i
        end
    end
    
    # 3. Construct the resulting string and a closure for evaluation
    c = best_coeffs[1]
    name = template_names[best_template_idx]
    expr_str = Printf.@sprintf("V(s) = %.4f * %s", c, name)
    
    return expr_str
end

"""
    evaluate_expression(expr_str, s)

Evaluates the distilled symbolic string. 
In this refined version, we use the known templates to parse the string back.
"""
function evaluate_expression(expr_str, s)
    # Extract coefficient and template from string
    # V(s) = 0.4800 * (exp(s)+exp(-s)-2)
    m = match(r"V\(s\) = ([\d\.\-]+) \* (.+)", expr_str)
    if m === nothing return 0.0f0 end
    
    coeff = parse(Float32, m.captures[1])
    tmpl_name = m.captures[2]
    
    if tmpl_name == "s^2"
        return coeff * s^2
    elseif tmpl_name == "(exp(s)+exp(-s)-2)"
        return coeff * (exp(s) + exp(-s) - 2.0f0)
    elseif tmpl_name == "log(cosh(s))"
        return coeff * log(cosh(s))
    elseif tmpl_name == "|s|"
        return coeff * abs(s)
    else
        return 0.0f0
    end
end

end # module
