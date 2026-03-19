module SpectralUtils

using Flux
using FFTW
using AbstractFFTs
using LinearAlgebra

export SpectralLayer, GatedSpectralLayer

"""
    spectral_init(in, out, modes)

Spectral Xavier Initialization.
Scales weights symmetrically in the complex domain.
"""
function spectral_init(in_channels::Int, out_channels::Int, modes::Int)
    scale = (1.0f0 / (in_channels * out_channels))
    # weights: (modes, out, in)
    w = complex.(randn(Float32, modes, out_channels, in_channels), 
                 randn(Float32, modes, out_channels, in_channels)) .* scale
    return w
end

"""
    SpectralLayer(in_channels, out_channels, modes)

A Fourier Neural Operator (FNO) spectral layer.
"""
struct SpectralLayer
    weights::AbstractArray{ComplexF32, 3}
    in_channels::Int
    out_channels::Int
    modes::Int
end

function SpectralLayer(in_channels::Int, out_channels::Int, modes::Int)
    weights = spectral_init(in_channels, out_channels, modes)
    return SpectralLayer(weights, in_channels, out_channels, modes)
end

Flux.@layer SpectralLayer

function (m::SpectralLayer)(x::AbstractArray{Float32})
    # Ensure 2D (features, batch)
    x_2d = ndims(x) == 1 ? reshape(x, :, 1) : x
    x_dense = collect(x_2d)
    x_ft = rfft(x_dense, 1) 
    
    # 2. Filter modes
    # x_ft shape: (modes_available, batch)
    n_modes = size(x_ft, 1)
    modes_to_keep = min(m.modes, n_modes)
    
    # 3. Complex multiplication (Simplified 1D)
    # We slice without mutation
    y_ft_clipped = x_ft[1:modes_to_keep, :] .* m.weights[1:modes_to_keep, 1, 1]
    
    # 4. Non-mutating padding
    # Padding back to original FFT size via vcat
    pad_size = n_modes - modes_to_keep
    if pad_size > 0
        y_ft = vcat(y_ft_clipped, zeros(ComplexF32, pad_size, size(x, 2)))
    else
        y_ft = y_ft_clipped
    end
    
    # 5. Inverse FFT
    return irfft(y_ft, size(x, 1), 1)
end

"""
    GatedSpectralLayer(in, out, modes)

Advanced FNO block with a Gated-Residual bypass.
Path A: Spectral Filter (Global)
Path B: Linear Shift (Local)
Combined via addition or gating.
"""
struct GatedSpectralLayer
    spectral::SpectralLayer
    spatial::Dense
end

function GatedSpectralLayer(in_channels::Int, out_channels::Int, modes::Int)
    return GatedSpectralLayer(
        SpectralLayer(in_channels, out_channels, modes),
        Dense(in_channels, out_channels)
    )
end

Flux.@layer GatedSpectralLayer

function (m::GatedSpectralLayer)(x::AbstractArray{Float32, 2})
    # Dual-Path Fusion
    out_spectral = m.spectral(x)
    out_spatial  = m.spatial(collect(x))
    
    # We use a residual-style addition
    # This ensures that local details (spatial) are preserved 
    # while global flow (spectral) is learned.
    return out_spectral .+ out_spatial
end

end # module
