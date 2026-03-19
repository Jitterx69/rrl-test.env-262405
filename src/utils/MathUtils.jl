module MathUtils

export compute_returns, compute_advantages

"""
    compute_returns(rewards, gamma=0.99)
Standard discounted return calculation.
"""
function compute_returns(rewards, gamma=0.99)
    returns = similar(rewards, Float32)
    G = 0.0f0
    for t in reverse(1:length(rewards))
        G = Float32(rewards[t]) + Float32(gamma) * G
        returns[t] = G
    end
    return returns
end

# Legacy alias for backward compatibility with older scripts
const compute_advantages = compute_returns

end # module
