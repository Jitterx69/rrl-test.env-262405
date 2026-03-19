using Pkg
Pkg.activate(".")
using Flux, Zygote, Functors

struct TestStruct
    m
end
# Try explicit functor if @layer failed previously
Functors.@functor TestStruct

function test_zygote_struct()
    println("--- Zygote Struct Test ---")
    m = Chain(Dense(1, 1))
    s = TestStruct(m)
    
    try
        gs = Zygote.gradient(s) do obj
            sum(obj.m([1.0f0]))
        end
        println("SUCCESS: Gradient is ", gs)
    catch e
        println("FAILURE: ", e)
    end
end

test_zygote_struct()
