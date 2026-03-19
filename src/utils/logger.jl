using CSV, DataFrames, Dates

mutable struct ExperimentLogger
    metrics::DataFrame
    trajectory::DataFrame
    path::String
end

function init_logger(path::String)

    mkpath(path)

    metrics = DataFrame(
        episode=Int[],
        reward=Float64[],
        stability=Float64[]
    )

    trajectory = DataFrame(
        timestep=Int[],
        state=Float64[],
        action=Float64[],
        prediction=Float64[],
        next_state=Float64[]
    )

    return ExperimentLogger(metrics, trajectory, path)
end

function log_metrics!(logger, ep, reward, stability)
    push!(logger.metrics, (ep, reward, stability))
end

function log_trajectory!(logger, t, s, a, r, s_next)
    push!(logger.trajectory, (t, s, a, r, s_next))
end

function finalize_logger!(logger)

    CSV.write(joinpath(logger.path, "metrics.csv"), logger.metrics)
    CSV.write(joinpath(logger.path, "trajectory.csv"), logger.trajectory)

    open(joinpath(logger.path, "metadata.txt"), "w") do io
        write(io, "Generated at: $(now())\n")
    end
end