using AAP

args = getArguments()
N = args[:N]
Re = 5000.0
g, uh, dofs_u, dofs_p = ResidualNS(N; μ=1/Re)

Ndofs = size(uh,1)
# FieldWise renumbering is fundamental for good performance and correctness
dofs_u = minimum(dofs_u):maximum(dofs_u)
dofs_p = minimum(dofs_p):maximum(dofs_p)

mask_type = args[:mask]
if mask_type == :p
    mask = dofs_p
elseif mask_type == :u
    mask = dofs_u
else # == :none
    mask = nothing
end

t0 = time()
out  = solveNonlinear(g, uh, args; mask=mask, mode=:picard)
sol_time = time() - t0

Ndofs = out[1]
its = out[2]
args[:solver] == "aap" && display(AAP.to)
println("DOFS: $(Ndofs), u=$(length(dofs_u)), p=$(length(dofs_p))")
println("Iterations:", its)
println("Solution time:", sol_time)
