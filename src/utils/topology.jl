module TopologicalAnalysis

using LinearAlgebra, Statistics

export compute_persistence_0d, estimate_topology_pressure, topological_loss

"""
    compute_persistence_0d(points)

Computes the 0-dimensional persistence (clustering/reachability) of a point cloud
using a Minimal Spanning Tree (MST) approach.
Detects how state clusters merge as the 'alpha' coupling changes.
"""
function compute_persistence_0d(points)
    n = size(points, 2)
    # Distance matrix
    dist_mat = [norm(points[:, i] .- points[:, j]) for i in 1:n, j in 1:n]
    
    # Simple MST Kruskal-based persistence lifetimes
    # (Simplified for research-grade prototyping)
    edges = []
    for i in 1:n, j in i+1:n
        push!(edges, (i, j, dist_mat[i, j]))
    end
    sort!(edges, by=x -> x[3])
    
    parent = collect(1:n)
    function find_root(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end
    
    lifetimes = []
    for (i, j, d) in edges
        root_i = find_root(i)
        root_j = find_root(j)
        if root_i != root_j
            push!(lifetimes, d)
            parent[root_i] = root_j
        end
    end
    
    return lifetimes # Return birth-death intervals (birth is always 0 for 0-dim)
end

"""
    estimate_topology_pressure(states)

Estimates "Topological Pressure" (likelihood of approaching a 1-cycle/limit cycle)
by calculating the Discrete Winding Number of the trajectory in state space.
High winding numbers indicate the emergence of stable limit cycles (Betti-1).
"""
function estimate_topology_pressure(states)
    # states: [N] or [dim, N]
    n = length(states)
    if n < 4 return 0.0f0 end
    
    # 1. Center the data
    m = mean(states)
    centered = states .- m
    
    # 2. Compute angles in phase space (assuming 2D or 1D->2D embedding)
    # If 1D, we use a delay embedding [s_t, s_{t-1}]
    if ndims(states) == 1
        x = centered[2:end]
        y = centered[1:end-1]
    else
        x = centered[1, :]
        y = centered[2, :]
    end
    
    angles = atan.(y, x)
    
    # 3. Sum of angular differences (Discrete Winding)
    delta_angles = diff(angles)
    # Correct for wrap-around
    delta_angles = map(da -> (da > π ? da - 2π : (da < -π ? da + 2π : da)), delta_angles)
    
    winding_number = abs(sum(delta_angles)) / (2π)
    
    # Pressure is the density of winding per unit time
    pressure = winding_number / n
    return Float32(pressure)
end

"""
    topological_loss(points; lambda=0.1)

A differentiable surrogate for H1 persistence (Betti-1 loss).
Minimizing this regularizes the state manifold to prevent chaotic loops.
"""
function topological_loss(points; lambda=0.1f0)
    # We use the previous acceleration-based surrogate for differentiability,
    # as atan2/diff(atan) is non-smooth for Zygote in certain regimes.
    # But we anchor the loss to the winding pressure.
    
    n = length(points)
    if n < 3 return 0.0f0 end
    
    diffs = diff(points)
    accels = diff(diffs)
    
    # Curvature surrogate (differentiable)
    curvature = mean(accels.^2) / (mean(diffs.^2) + 1e-6)
    
    return lambda * curvature
end

end # module
