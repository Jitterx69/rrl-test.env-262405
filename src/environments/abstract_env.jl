abstract type AbstractEnv end

function reset!(env::AbstractEnv)
    error("reset! not implemented")
end

function step!(env::AbstractEnv, s, a)
    error("step! not implemented")
end

function reward(env::AbstractEnv, s, a, s_next)
    error("reward not implemented")
end