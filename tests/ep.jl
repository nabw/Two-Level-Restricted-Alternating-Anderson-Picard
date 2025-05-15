using AAP
args = getArguments()

N = args[:N]
ff, u, dofs_e, dofs_i = ResidualEP(N)
dofs = length(u)
dofs_e = minimum(dofs_e):maximum(dofs_e)
dofs_i = minimum(dofs_i):maximum(dofs_i)

mask_type = args[:mask]
if mask_type == :i
    mask = dofs_i
elseif mask_type == :e
    mask = dofs_e
else # == :none
    mask = nothing
end


t0 = time()
out = solveNonlinear(ff, u, args; mask)
t_solve = time() - t0

dofs = out[1]
its = out[2]
println(AAP.to)
println("Dofs:", dofs)
println("Iterations:", its)
println("Solution time:", t_solve)
