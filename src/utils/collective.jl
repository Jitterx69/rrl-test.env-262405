module SuperRadiance

using LinearAlgebra, Statistics, FFTW

export CoherentSpectralLayer, compute_population_entropy, critical_slowing_index

"""
    CoherentSpectralLayer(N; bandwidth=0.1)

A global aggregation layer that find dominant synchronization modes 
across a population of N agents using FFT.
"""
function CoherentSpectralLayer(signals; bandwidth=0.1f0)
    N = length(signals)
    # 1. FFT of agent signals across the population domain
    freq_data = fft(signals)
    
    # 2. Extract collective magnitude (Energy in the DC and low-freq components)
    collective_energy = abs.(freq_data)
    
    # 3. Filter for coherence (Information Super-radiance)
    # Thresholding to find "Lasing" modes
    lasing_modes = collective_energy .> (mean(collective_energy) + 2.0 * std(collective_energy))
    
    return Float32(mean(collective_energy[lasing_modes] .+ 1f-6))
end

"""
    compute_population_entropy(signals)

Computes the Shannon Entropy of the population signal distribution.
A sudden drop in entropy indicates the "Super-radiance" phase transition.
"""
function compute_population_entropy(signals)
    # Bin the signals to compute distribution
    bins = range(minimum(signals), maximum(signals), length=30)
    counts = [count(x -> (x >= bins[i] && x < bins[i+1]), signals) for i in 1:(length(bins)-1)]
    p = counts ./ (sum(counts) + 1e-8)
    entropy = -sum(p .* log2.(p .+ 1e-8))
    return Float32(entropy)
end

"""
    critical_slowing_index(history)

Detects 'Critical Slowing Down' (CSD) by measuring temporal autocorrelation.
High CSD indicates the population is approaching a tipping point (Super-radiance).
"""
function critical_slowing_index(history)
    if length(history) < 10 return 0.0f0 end
    # Temporal variance recovery proxy
    v = var(history)
    ac = mean(history[1:end-1] .* history[2:end]) / (v + 1e-6)
    return Float32(abs(ac))
end

end # module
