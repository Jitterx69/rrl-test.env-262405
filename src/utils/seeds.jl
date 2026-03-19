using Random

function set_seed(seed::Int)
    Random.seed!(seed)
end