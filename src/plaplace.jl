function ResidualPLaplace(N; p=1.5, β=1e1)
    ## Generate a grid
    L = 1.0
    dim = 3
    model = CartesianDiscreteModel((0,L,0,L,0,L),(N,N,N))

    ## Material parameters
    labels = get_face_labeling(model)

    order = 1
    reffe = ReferenceFE(lagrangian,Float64,order)
    V0 = TestFESpace(model,reffe,conformity=:H1,labels=labels,dirichlet_tags=["boundary"])
    Ug = TrialFESpace(V0,[0])
    degree=2
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    flux(∇u) = norm(∇u)^(p-2) * ∇u
    f(x) = 1
    res(u,v) = ∫( ∇(v)⊙(flux∘∇(u)) - v*f)*dΩ
    dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⊙∇du)*∇u+norm(∇u)^(p-2)*∇du
    jac(u,du,v) = ∫( ∇(v)⊙(dflux∘(∇(du),∇(u))) + β*(∇(du)⋅∇(v) + du*v))*dΩ
    #jac(u,du,v) = ∫(∇(v)⊙(dflux∘(∇(du),∇(u))))*dΩ
    #jac(u,du,v) = ∫( ∇(du)⋅∇(v))*dΩ
    op = FEOperator(res,jac,Ug,V0)


    aa(u,v) = ∫(∇(v)⊙∇(u))*dΩ
    ll(v) = ∫(v*f)*dΩ
    uh = zero(Ug)
    x = copy(uh.free_values)


    # Solve harmonic problem for initialization
    A0,f0=assemble_matrix_and_vector(aa,ll,Ug,V0)
    options = "-ksp_type cg -pc_type gamg"
    PetscCall.init(args=split(options))
    setup = PetscCall.ksp_setup(x,A0,f0)
    results = PetscCall.ksp_solve!(x,setup,f0)
    uh.free_values .= x
    PetscCall.ksp_finalize!(setup)

    b, A = Gridap.Algebra.residual_and_jacobian(op, uh)
    #prec = aspreconditioner(ruge_stuben(A))
    options = "-ksp_type preonly -pc_type gamg"
    PetscCall.init(args=split(options))
    prec = PetscCallPreconditioner(uh.free_values, A, b)


    function ff(f, x)
        uh.free_values .= x
        #Gridap.Algebra.residual_and_jacobian!(b, A, op, uh)
        #prec = aspreconditioner(ruge_stuben(A)) 
        Gridap.Algebra.residual!(b, op, uh)
        rmul!(b,-1)
        ldiv!(f,prec,b)
    end
    return ff, copy(uh.free_values), prec
end
