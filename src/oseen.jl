function assembleOseen(N; μ=1e-1)
    # Parameters
    # Zero advection to do Stokes
    BB = VectorValue(0.0,0.0,0.0)
    
    domain = (0,1,0,1,0,1)
    partition = (N,N,N)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"y0",[1,2,5,6,9 ,11,17,18,23])
    add_tag_from_tags!(labels,"yL",[3,4,7,8,10,12,19,20,24])
    add_tag_from_tags!(labels,"z0",[1,2,3,4,9 ,10,13,14,21])
    add_tag_from_tags!(labels,"zL",[5,6,7,8,11,12,15,16,22])

    order = 2
    reffeᵤ = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
    reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
    V = TestFESpace(model,reffeᵤ,labels=labels,dirichlet_tags=["y0","yL","z0","zL"],conformity=:H1)
    Q = TestFESpace(model,reffeₚ,conformity=:H1)
    Y = MultiFieldFESpace([V,Q])
    u0 = VectorValue(0.,0.,0.)
    U = TrialFESpace(V,[u0,u0,u0,u0])
    Q = TrialFESpace(Q)
    X = MultiFieldFESpace([U,Q])
    degree = order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ,degree)
    f = VectorValue(1.0,1.0,1.0)
    a((u,p),(v,q)) = ∫( μ * ∇(v)⊙∇(u) + (∇(u)⋅BB)⋅v- (∇⋅v)*p - q*(∇⋅u) )dΩ
    l((v,q)) = ∫( v⋅f )dΩ
    
    p((u,p),(v,q)) = ∫(μ * ∇(v)⊙∇(u) + (∇(u)⋅BB)⋅v + 1.0/μ * p*q)*dΩ
    
    K, f = assemble_matrix_and_vector(a,l,X,Y)
    Mp = assemble_matrix(p,X,Y)
    Ndofs = length(f)
    
    # Fields are naturally in order
    dofs_f = get_free_dof_ids(X.spaces[1])
    dofs_p = get_free_dof_ids(X.spaces[2])
    
    # Shift accordingly to obtain masks
    dofs_f = (minimum(dofs_f)):(maximum(dofs_f))
    len_f = length(dofs_f)
    dofs_p = (minimum(dofs_p)+len_f):(maximum(dofs_p)+len_f)
    
    return K, Mp, f, dofs_f, dofs_p
end


