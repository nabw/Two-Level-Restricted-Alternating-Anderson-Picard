using AAP
args = getArguments()

N = args[:N]
ff, u = getResidualFunction(N)
dofs = length(u)

t0 = time()
out = solveNonlinear(ff, u, args)
t_solve = time() - t0

dofs = out[1]
its = out[2]
println(AAP.to)
println("Dofs:", dofs)
println("Iterations:", its)
println("Solution time:", t_solve)
