using PetscCall, AAP

args = getArguments()
N = args[:N]
K, Mp, f, dofs_u, dofs_p = assembleOseen(N; μ=1e-1)

Ndofs = size(f,1)
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

options = "-ksp_type preonly 
            -pc_hypre_boomeramg_interp_type ext+i
            -pc_hypre_boomeramg_coarsen_type HMIS
            -pc_type hypre"
PetscCall.init(args=split(options))

u = zeros(Ndofs)
prec = PetscCallPreconditioner(u, Mp, f)

t0 = time()
out = solveLinear(K, f, u, prec, args; mask=mask)
sol_time = time() - t0

its = out[2]
args[:solver] == "aap" && display(AAP.to)
println("DOFS: $(Ndofs), u=$(length(dofs_u)), p=$(length(dofs_p))")
println("Iterations:", its)
println("Solution time:", sol_time)
