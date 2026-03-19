module Architectures

using Flux, Functors, Optimisers
include("spectral_utils.jl")
using .SpectralUtils

export ReflexiveOracle, GaussianPolicy, SpectralOracle, GatedSpectralOracle

struct ReflexiveOracle
    model
end
Flux.@layer ReflexiveOracle

function ReflexiveOracle(in_dim::Int, out_dim::Int=1)
    m = Chain(Dense(in_dim, 64, relu), Dense(64, 64, relu), Dense(64, out_dim))
    return ReflexiveOracle(m)
end

function (o::ReflexiveOracle)(x)
    # Ensure x is an array for Flux layers
    x_input = x isa Number ? [Float32(x)] : x
    return o.model(x_input)
end

# =========================================================
# 3. Spectral Oracle (FNO-based for Discretization Invariance)
# =========================================================

"""
    SpectralOracle(in_dim, out_dim, hidden_dim, modes)

A high-tech reflexive oracle using Fourier spectral layers.
Captures global system dynamics in the frequency domain.
"""
struct SpectralOracle
    model::Chain
end

function SpectralOracle(in_dim::Int, out_dim::Int, hidden_dim::Int=64, modes::Int=16)
    model = Chain(
        Dense(in_dim, hidden_dim, relu),
        # Treat the hidden features as a 1D signal for spectral filtering
        SpectralLayer(hidden_dim, hidden_dim, modes),
        Dense(hidden_dim, out_dim)
    )
    return SpectralOracle(model)
end

function (m::SpectralOracle)(x)
    x_input = x isa Number ? [Float32(x)] : x
    return m.model(x_input)
end

# =========================================================
# 4. Gated Spectral Oracle (FNO+ with Residual Bypass)
# =========================================================

"""
    GatedSpectralOracle(in_dim, out_dim, hidden_dim, modes)

State-of-the-art spectral oracle using Gated Fourier Units.
Combines global spectral filtering with local spatial dense paths.
"""
struct GatedSpectralOracle
    model::Chain
end

function GatedSpectralOracle(in_dim::Int, out_dim::Int, hidden_dim::Int=64, modes::Int=16)
    model = Chain(
        Dense(in_dim, hidden_dim, relu),
        GatedSpectralLayer(hidden_dim, hidden_dim, modes),
        relu,
        Dense(hidden_dim, out_dim)
    )
    return GatedSpectralOracle(model)
end

Flux.@layer GatedSpectralOracle

function (m::GatedSpectralOracle)(x)
    x_input = x isa Number ? [Float32(x)] : x
    return m.model(x_input)
end

# =========================================================
# 5. Stochastic Policy (Gaussian)
# =========================================================

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
    x_input = x isa Number ? [Float32(x)] : x
    mv_out = p.mu_net(x_input)
    # Clip log_sigma to prevent NaN in exp
    sv_out = exp.(clamp.(p.log_sigma, -5.0f0, 2.0f0))
    return mv_out, sv_out
end

end # module
