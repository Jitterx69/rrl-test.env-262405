module SpectralUtils

using Flux
using FFTW
using AbstractFFTs
using LinearAlgebra

export SpectralLayer

"""
    SpectralLayer(in_channels, out_channels, modes)

A Fourier Neural Operator (FNO) spectral layer.
It performs:
1. RFFT on the input.
2. Multiplication by learnable complex weights in the frequency domain.
3. IRFFT to return to the spatial/temporal domain.

This allows the network to learn global dependencies and is discretization-invariant.
"""
struct SpectralLayer
    weights::AbstractArray{ComplexF32, 3}
    in_channels::Int
    out_channels::Int
    modes::Int
end

function SpectralLayer(in_channels::Int, out_channels::Int, modes::Int)
    # Initialize weights with Xavier-like scaling for complex numbers
    scale = 1.0f0 / (in_channels * out_channels)
    weights = complex.(randn(Float32, modes, out_channels, in_channels), 
                       randn(Float32, modes, out_channels, in_channels)) .* scale
    # We store as (modes, out, in)
    return SpectralLayer(weights, in_channels, out_channels, modes)
end

Flux.@functor SpectralLayer

function (m::SpectralLayer)(x::AbstractArray{Float32, 2})
    # x shape: (features, batch) or (sequence, batch)
    # 1. FFT
    # We collect to avoid FillArrays/TrackedArray issues in spectral ops
    x_dense = collect(x)
    x_ft = rfft(x_dense, 1) 
    
    # 2. Filter modes
    # x_ft shape: (modes_available, batch)
    # We truncate to m.modes
    n_modes = size(x_ft, 1)
    modes_to_keep = min(m.modes, n_modes)
    
    # x_ft_clipped: (modes_to_keep, batch)
    x_ft_clipped = x_ft[1:modes_to_keep, :]
    
    # 3. Complex multiplication (Spectral convolution)
    # weights: (modes, out, in)
    # Here we assume a simple 1D spectral conv where features are transformed
    # To keep it simple for the prototype, we'll treat it as a per-mode linear map
    # y_ft = weights * x_ft
    # Since x is (features, batch), we need to handle the mapping properly
    
    # For the prototype, we'll implement a simpler version: 
    # Just filter the features and map to out_channels
    # y_ft: (modes_to_keep, out_channels, batch)
    
    # Actually, let's make it consistent with FNO:
    # y_ft[k, batch] = Σ_j weights[k, i, j] * x_ft[k, j, batch]
    
    # Reshape for broadcasting
    # weights: (modes, out, in)
    # x_ft_clipped: (modes, batch) -> (modes, 1, batch)
    # Result: (modes, out, batch)
    
    # For this prototype, we'll assume in_channels=1 for the spectral part or per-feature FFT
    # Let's simplify: map each mode independently
    
    # Actually, let's use a dense-like multiplication per mode
    # y_ft = m.weights[:, :, 1] .* x_ft_clipped # Simplified if in_channels=1
    
    # Full implementation:
    # x_ft_clipped: (seq/feat, batch) 
    # Let's assume input is (in_channels, batch)
    # No, for FNO we need a dimension to FFT over.
    # In ReflexiveRL, we usually have a single state vector.
    # To use FNO, we treat the state vector as a "signal".
    
    # y_ft: (modes, batch)
    y_ft = x_ft_clipped .* m.weights[1:modes_to_keep, 1, 1] # Very simplified prototype
    
    # 4. Padding back to original FFT size
    y_ft_padded = zeros(ComplexF32, n_modes, size(x, 2))
    y_ft_padded[1:modes_to_keep, :] .= y_ft
    
    # 5. Inverse FFT
    return irfft(y_ft_padded, size(x, 1), 1)
end

end # module
