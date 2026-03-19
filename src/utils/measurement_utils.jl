module MeasurementUtils

using LinearAlgebra, Statistics

export reflexive_consistency_error, feedback_sensitivity, aggregate_sweep_metrics

"""
    reflexive_consistency_error(s_t, s_t_next)
Calculates the Euclidean distance between successive states as a proxy 
for the reflexive operator residual ||s - Phi(s)||.
"""
function reflexive_consistency_error(s_t, s_t_next)
    return norm(s_t .- s_t_next)
end

"""
    feedback_sensitivity(env, s, a, r_p)
Approximates the influence of the predictive signal on the environment 
transition: Delta s = f(s, a(r_p), r_p) - f(s, a(0), 0).
"""
function feedback_sensitivity(env, s, a, r_p)
    # This assumes we can query the environment or simulate the transition
    # For a simple scalar environment like Tier 1:
    # Delta s = (s + a - alpha*r_p) - (s + a - alpha*0) = -alpha * r_p
    # In general, we'd need to simulate both pathways if possible.
    return abs(env.alpha * r_p) 
end

"""
    aggregate_sweep_metrics(results_df)
Aggregates metrics across seeds and alpha points to identify stability regimes.
"""
function aggregate_sweep_metrics(results; groupby_cols=[:alpha, :algo])
    # Placeholder for more complex statistical analysis
    return results
end

end # module
