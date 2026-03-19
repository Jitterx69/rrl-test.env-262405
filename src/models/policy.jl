using Flux

struct Policy
    model
end

function Policy(hidden::Int)
    m = Chain(
        Dense(1, hidden, relu),
        Dense(hidden, 1)
    )
    return Policy(m)
end

Flux.@functor Policy

(p::Policy)(r) = p.model([r])[1]