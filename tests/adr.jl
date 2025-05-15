using PetscCall, AAP
dim = 3
args = getArguments()
 
N = args[:N]
K, f = assembleADR(dim, N)

Ndofs = size(K, 1)
options = "-ksp_type preonly -pc_type ilu -pc_factor_levels 2"
PetscCall.init(args=split(options))
u = zeros(Ndofs)
Prec = PetscCallPreconditioner(u, K, f)

# Warm-up
out = solveLinear(K, f, u, Prec, args)
u .= 0.0
t0 = time()
out = solveLinear(K, f, u, Prec, args)
sol_time = time() - t0

its = out[2]
args[:solver] == "aap" && display(AAP.to)
println("Dofs:", Ndofs)
println("Iterations:", its)
println("Solution time:", sol_time)

