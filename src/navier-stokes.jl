function ResidualNS(N; μ=1e-1, dim=2)
    if dim == 2
        partition = (N, N)
        domain = (0,1,0,1)
    else
        partition = (N,N,N)
        domain = (0,1,0,1,0,1)
    end
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    labels = get_face_labeling(model)
    if dim == 3
        add_tag_from_tags!(labels,"x0",[1,3,5,7,13,15,17,19,25])
        add_tag_from_tags!(labels,"xL",[2,4,6,8,14,16,18,20,26])
        add_tag_from_tags!(labels,"y0",[1,2,5,6,9 ,11,17,18,23])
        add_tag_from_tags!(labels,"yL",[3,4,7,8,10,12,19,20,24])
        add_tag_from_tags!(labels,"z0",[1,2,3,4,9 ,10,13,14,21])
        add_tag_from_tags!(labels,"zL",[5,6,7,8,11,12,15,16,22])
        dirichlet_tags = ["x0","xL","y0","yL","z0","zL"]
        doall = (true, true, true)
        dirichlet_masks = [doall, doall, doall, 
                           doall, doall, (true,true,true)]
        u0 = VectorValue(0.,0.,0.)
        u_bc = VectorValue(1.,0.,0.)
        bcs = [u0,u0,u0,u0,u0,u_bc]
        ff = VectorValue(0.,0.,0.)
    else
        add_tag_from_tags!(labels,"x0",[1,3,7])
        add_tag_from_tags!(labels,"xL",[2,4,8])
        add_tag_from_tags!(labels,"y0",[1,2,5])
        add_tag_from_tags!(labels,"yL",[3,4,6])
        dirichlet_tags = ["x0","xL","y0","yL"]
        doall = (true, true)
        dirichlet_masks = [doall, doall,
                           doall, (true,true)]
        u0 = VectorValue(0.,0.)
        u_bc = VectorValue(1.,0.)
        bcs = [u0,u0,u0,u_bc]
        ff = VectorValue(0.,0.)
    end

    order = 2
    reffeᵤ = ReferenceFE(lagrangian,VectorValue{dim,Float64},order)
    reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
    V = TestFESpace(model,reffeᵤ,labels=labels,
                    dirichlet_tags=dirichlet_tags,
                    conformity=:H1, 
                    dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model,reffeₚ,conformity=:H1, constraint=:zeromean)
    #Q = TestFESpace(model,reffeₚ,conformity=:H1)
    Y = MultiFieldFESpace([V,Q])
    U = TrialFESpace(V,bcs)
    Q = TrialFESpace(Q)
    X = MultiFieldFESpace([U,Q])
    degree = order
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    wh = zero(U)
    solh = zero(X)
    # We implement advecetion as Rebholz et al.
    #b(u,v,w) = 0.5 * ( ((∇(v)')⋅u)⋅w - ((∇(w)')⋅u)⋅v )
    b(w,u,v) = v⋅((∇(u)')⋅w)
    γ = 1.
    a(wh, (u,p),(v,q)) = ∫(μ * ∇(v)⊙∇(u) + γ*(∇⋅u)*(∇⋅v)
                           + b(wh,u,v) - (∇⋅v)*p + q*(∇⋅u) )dΩ
    l((v,q)) = ∫( v⋅ff )dΩ
    
    Ndofs = length(wh.free_values)

    
    # Fields are naturally in order
    dofs_f = get_free_dof_ids(X.spaces[1])
    dofs_p = get_free_dof_ids(X.spaces[2])
    
    # Shift accordingly to obtain masks
    dofs_f = (minimum(dofs_f)):(maximum(dofs_f))
    len_f = length(dofs_f)
    dofs_p = (minimum(dofs_p)+len_f):(maximum(dofs_p)+len_f)
    

    cache = nothing
    function g(g,x,ω)
        solh.free_values .= x
        sol_u, sol_p = solh
        wh.free_values .= sol_u.free_values
        ak(xx,yy) = a(wh, xx, yy)
        op = AffineFEOperator(ak,l,X,Y)
        ls = LUSolver()
        solver = LinearFESolver(ls)
        solh, cache = Gridap.solve!(solh,solver,op, cache)
        g .= solh.free_values
    end

    return g, copy(solh.free_values), dofs_f, dofs_p
end


