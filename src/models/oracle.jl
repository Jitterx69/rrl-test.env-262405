using Flux

struct Oracle
    model
end

function Oracle(input_dim::Int, hidden::Int)
    m = Chain(
        Dense(input_dim, hidden, relu),
        Dense(hidden, 1)
    )
    return Oracle(m)
end

Flux.@functor Oracle

(o::Oracle)(s) = o.model([s])[1]