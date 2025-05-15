using AAP
args = getArguments()

p = 1.5 # Exponent
β = 1e1 # LM stabilization
N = args[:N]
ff, u = ResidualPLaplace(N; p=p, β=β)
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
