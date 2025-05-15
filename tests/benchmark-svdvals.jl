using RandomizedLinAlg
using TSVD
using LinearAlgebra
using TimerOutputs

function invPowerIteration(A; Nits=1, verbose=false)
    val = 0

    lA = LowerTriangular(A)
    uA = UpperTriangular(A')
    
    x = rand(size(A,1))
    y = similar(x)
    u = similar(x)
    for i in 1:Nits
        y .= uA\x
        x .= lA\y
        if verbose
            u .= (A'A)*x
            val = dot(x, u) / dot(x,x)
            println("DEBUG: $(val)")
        end
    end
    u .= (A'A)*x
    val = dot(x, u) / dot(x,x)
    return sqrt(val)
end

Ns = [100,200,400,800,1600,3200]
#Ns = [100,200,400]
for N in Ns
    println("Computing $(N)")
    A = ones(N, N)
    C = A'A
    A = tril(C)
    B = copy(C)
    @time "Standard" svdvals(A)
    #@time "Standard!" svdvals!(B) # very similar times
    #@time "TSVD" tsvd(A,N) # Doesnt converge
    #@time "Randomized" RandomizedLinAlg.rsvdvals(A,N) # Much slower
    @time "InvPower" invPowerIteration(A)
end

A = tril(ones(1000,1000))
val = minimum(svdvals(A))
nval = invPowerIteration(A; Nits=20, verbose=true)
println("Computed: $(val), Approx: $(nval)")
