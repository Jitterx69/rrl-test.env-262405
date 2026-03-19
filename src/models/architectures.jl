module Architectures

using Flux, Functors, Optimisers

export ReflexiveOracle, GaussianPolicy

struct ReflexiveOracle
    model
end
Flux.@layer ReflexiveOracle

function ReflexiveOracle(in_dim::Int, out_dim::Int=1)
    m = Chain(Dense(in_dim, 64, relu), Dense(64, 64, relu), Dense(64, out_dim))
    ReflexiveOracle(m)
end

function (o::ReflexiveOracle)(x)
    return o.model(x)
end

struct GaussianPolicy
    mu_net
    log_sigma
end
Flux.@layer GaussianPolicy

function GaussianPolicy(in_dim::Int, out_dim::Int=1)
    m_net = Chain(Dense(in_dim, 64, relu), Dense(64, 64, relu), Dense(64, out_dim))
    ls = fill(-2.0f0, out_dim)
    GaussianPolicy(m_net, ls)
end

function (p::GaussianPolicy)(x)
    mv_out = p.mu_net(x)
    # Clip log_sigma to prevent NaN in exp
    sv_out = exp.(clamp.(p.log_sigma, -5.0f0, 2.0f0))
    return mv_out, sv_out
end

end # module
