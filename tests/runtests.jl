using AAP, PetscCall
using Test

# Default args for tests
args = getArguments()
args[:maxit] = 10 # We only test that they run

@testset "Poisson" begin
    for dim in (2,3)
        N = args[:N]
        K, f = assemblePoisson(dim, N)

        opts = "-ksp_type preonly -pc_type ilu -pc_factor_levels 2"
        PetscCall.init(args=split(opts))
        u = similar(f)
        u .= 0.0
        Prec = PetscCallPreconditioner(u, K, f)
        solveLinear(K, f, u, Prec, args)
    end
end


@testset "ADR" begin
    for dim in (2,3)
        N = args[:N]
        K, f = assembleADR(dim, N)

        opts = "-ksp_type preonly -pc_type ilu -pc_factor_levels 2"
        PetscCall.init(args=split(opts))
        u = similar(f)
        u .= 0.0
        Prec = PetscCallPreconditioner(u, K, f)
        solveLinear(K, f, u, Prec, args)
    end
end

@testset "Oseen" begin
    for mask_type in (:p, :u, :none)
        N = args[:N]
        K, Mp, f, dofs_u, dofs_p = assembleOseen(N; μ=1.0)

        Ndofs = size(f,1)
        # FieldWise renumbering is fundamental for good performance and correctness
        dofs_u = minimum(dofs_u):maximum(dofs_u)
        dofs_p = minimum(dofs_p):maximum(dofs_p)

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

        _ = solveLinear(K, f, u, prec, args; mask=mask)
    end
end

@testset "PLaplace" begin
    p = 1.5 # Exponent
    β = 1e1 # LM stabilization
    N = args[:N]
    ff, u = ResidualPLaplace(N; p=p, β=β)
    _ = solveNonlinear(ff, u, args)
end

@testset "Hyperelasticity" begin
    N = args[:N]
    ff, u = ResidualHyperelasticity(N)
    _ = solveNonlinear(ff, u, args)
end

@testset "Bidomain" begin
    for mask_type in (:e, :i, :none)
        N = args[:N]
        ff, u, dofs_e, dofs_i = ResidualEP(N)

        if mask_type == :e
            mask = dofs_e
        elseif mask_type == :i
            mask = dofs_i
        else # == :none
            mask = nothing
        end

        _ = solveNonlinear(ff, u, args; mask=mask)
    end
end

@testset "NavierStokes" begin
    for mask_type in (:p, :u, :none)
        N = args[:N]
        ff, u, dofs_u, dofs_p = ResidualNS(N)

        if mask_type == :p
            mask = dofs_p
        elseif mask_type == :u
            mask = dofs_u
        else # == :none
            mask = nothing
        end

        _ = solveNonlinear(ff, u, args; mask=mask, mode=:picard)
    end
end
