module MathUtils

export compute_returns

"""
    compute_returns(rewards, gamma=0.99)
Standard discounted return calculation.
"""
function compute_returns(rewards, gamma=0.99)
    returns = similar(rewards)
    G = 0.0
    for t in reverse(1:length(rewards))
        G = rewards[t] + gamma * G
        returns[t] = G
    end
    return returns
end

end # module
