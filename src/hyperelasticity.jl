
β = 1e0 # LM stab
struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    JC = det(C)
    J = sqrt(JC)
    Ciso = C * J^(-2.0/3.0)
    Ic = tr(Ciso)
    C1 = μ/2.0 
    D1 = λ/2.0 
    #return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
    return C1 * (Ic - 3) + D1 * (J-1) * log(J)
end

function constitutive_driver(C, mp::NeoHooke)
    ## Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

function assemble_residual!(ge, cell, cv, fv, mp, ue, ΓN)
    ## Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    tn = 0.1 # Traction (to be scaled with surface normal)
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ## Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
        ## Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

        ## Loop over test functions
        for i in 1:ndofs
            ## Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            ## Add contribution to the residual from this test function
            ge[i] += ∇δui ⊡ P * dΩ
        end
    end
end;


function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
    ## Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    tn = 0.1 # Traction (to be scaled with surface normal)
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ## Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
        ## Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

        ## Loop over test functions
        for i in 1:ndofs
            ## Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            ## Add contribution to the residual from this test function
            ge[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                δuj = shape_value(cv, qp, j)
                ∇δuj = shape_gradient(cv, qp, j)
                ## Add contribution to the tangent
                ke[i,j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
                ke[i,j] += β * (∇δui⊡ ∇δuj + δui ⋅ δuj) * dΩ # LM stab
            end
        end
    end
end;

# Assembling global residual and tangent is also done in the usual way, just looping over
# the elements, call the element routine and assemble in the the global matrix K and
# residual g.

function assemble_global_residual!(f, dh, cv, fv, mp, u, ΓN)
    n = ndofs_per_cell(dh)
    ge = zeros(n)

    ## start_assemble resets K and f
    #assembler = start_assemble(K, f, fillzero=false)
    f .= 0.0

    ## Loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        @timeit "element assemble" assemble_residual!(ge, cell, cv, fv, mp, ue, ΓN)
        assemble!(f, global_dofs, ge)
    end
end;


function assemble_global!(K, f, dh, cv, fv, mp, u, ΓN)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    ## start_assemble resets K and f
    assembler = start_assemble(K, f)

    ## Loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        @timeit "element assemble" assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
        assemble!(assembler, global_dofs, ke, ge)
    end
end;

function ResidualHyperelasticity(N)
    ## Generate a grid
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Tetrahedron, (N, N, N), left, right)

    ## Material parameters
    E = 10.0
    ν = 0.45
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    ## Finite element base
    ip = Lagrange{RefTetrahedron, 1}()^3
    qr = QuadratureRule{RefTetrahedron}(1)
    qr_facet = FacetQuadratureRule{RefTetrahedron}(1)
    cv = CellValues(qr, ip)
    fv = FacetValues(qr_facet, ip)

    ## DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, ip) # Add a displacement field
    close!(dh)

    function rotation(X, t, θ = deg2rad(30.0))
        x, y, z = X
        return t * Vec{3}(
            (0.0,
            L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ),
            L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ)
            ))
    end

    dbcs = ConstraintHandler(dh)
    ## Add a homogenous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfacetset(grid, "right"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfacetset(grid, "left"), (x,t) -> rotation(x, t), [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)
    t = 0.5
    Ferrite.update!(dbcs, t)

    ## Neumann part of the boundary
    ΓN = union(
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
        getfacetset(grid, "front"),
        getfacetset(grid, "back"),
    )

    _ndofs = ndofs(dh)
    u  = zeros(_ndofs)
    apply!(u, dbcs)

    ## Create sparse matrix and residual vector
    K = allocate_matrix(dh)
    g = zeros(_ndofs)

    # First global assembly to get preconditioner matrix
    assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
    apply!(K, g, dbcs)

    options = "-ksp_type preonly 
                -pc_type hypre"
    PetscCall.init(args=split(options))
    prec = PetscCallPreconditioner(u, K, g)

    temp = copy(u)
    temp .= 0.0
    function ff(g, x)
        # Right preconditioning
        ldiv!(temp, prec, x) # g <- P^{-1} g
        assemble_global_residual!(g, dh, cv, fv, mp, temp, ΓN)
        apply_zero!(g, dbcs)

        # Left preconditioning
        #assemble_global_residual!(temp, dh, cv, fv, mp, x, ΓN)
        #apply_zero!(temp, dbcs)
        #ldiv!(g, prec, temp) # temp <- P^{-1} g
        #axpby!(1.0, x, ω, g)
    end
    return ff, u, prec
end
