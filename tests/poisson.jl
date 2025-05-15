using Preconditioners, AAP

dim = 3
args = getArguments()

N = args[:N]
K, f = assemblePoisson(dim, N)

Ndofs = size(f,1)
Prec = CholeskyPreconditioner(K, 2)
u = zeros(Ndofs)

t0 = time()
out = solveLinear(K, f, u, Prec, args)
sol_time = time() - t0

its = out[2]
args[:solver] == "aap" && display(AAP.to)
println("Dofs:", Ndofs)
println("Iterations:", its)
println("Solution time:", sol_time)

