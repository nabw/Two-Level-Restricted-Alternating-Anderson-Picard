function ResidualEP(N)
    # Parameters
    dt = 1e-5 # ms
    
    L = 1e-2 # 1 cm
    domain = (0,L,0,L,0,L)
    partition = (N,N,N)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    labels = get_face_labeling(model)
    #add_tag_from_tags!(labels,"y0",[1,2,5,6,9 ,11,17,18,23])
    #add_tag_from_tags!(labels,"yL",[3,4,7,8,10,12,19,20,24])
    #add_tag_from_tags!(labels,"z0",[1,2,3,4,9 ,10,13,14,21])
    #add_tag_from_tags!(labels,"zL",[5,6,7,8,11,12,15,16,22])

    order = 1
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,labels=labels,conformity=:H1)
    X = MultiFieldFESpace([V,V])
    degree = order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ,degree)
    se = 0.3 
    si = 0.1
    Chi = 100.0
    cm = 10.0
    idt = 1.0 / dt
    v(xe,xi) = xe - xi
    r(x) = sqrt(x⋅x)
    x0 = VectorValue(0.0,0.0,0.0) # origin of slab
    v0 = 0.0
    Ia = 1e3
    Iapp(x) = r(x-x0) < 2e-3 ? Ia : 0.0
    b = 5 * 1e10
    c = 0.0001 # mV
    delta = 0.001
    beta = 10.
    #eta = 100000.0
    #gamma = 25.0
    # G = eta v - gamma w
    w = 0.0
    Iion(v, w) = -b * v * (v-c) * (delta - v) + beta * w

    # Closure function
    uh = zero(X)
    ue, ui = uh
    F((ue,ui), (ve,vi)) = ∫( Chi * cm * idt * (v(ue, ui)-v0) * v(ve,vi)
                           + se * ∇(ue)⋅∇(ve)
                           + si * ∇(ui)⋅∇(vi)
                           + Chi * Iion(v(ue,ui), w) * v(ve,vi)
                           - Iapp * v(ve,vi))dΩ
    
    Res((ve,vi)) = F((ue,ui), (ve,vi))
    res = assemble_vector(Res,X)
    Ndofs = length(uh.free_values)
    
    function ff(f, x)
        uh.free_values .= x
        ue, ui = uh
        Res((ve,vi)) = F((ue,ui), (ve,vi))
        Gridap.FESpaces.assemble_vector!(Res, res, X)
        rmul!(res,-1.0)
        copy!(f,res)
    end


    # Fields are naturally in order
    dofs_e = get_free_dof_ids(X.spaces[1])
    dofs_i = get_free_dof_ids(X.spaces[2])
    
    # Shift accordingly to obtain masks
    dofs_e = (minimum(dofs_e)):(maximum(dofs_e))
    len_e = length(dofs_e)
    dofs_i = (minimum(dofs_i)+len_e):(maximum(dofs_i)+len_e)
    
    return ff, copy(uh.free_values), dofs_e, dofs_i
end


