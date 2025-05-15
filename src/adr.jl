function assembleADR(dim, N)
    if dim == 3
        @timeit "create_grid" grid = generate_grid(Hexahedron, (N, N, N));
        ip = Lagrange{RefHexahedron, 1}()
        qr = QuadratureRule{RefHexahedron}(2)
    else
        @timeit "create_grid" grid = generate_grid(Quadrilateral, (N, N));
        ip = Lagrange{RefQuadrilateral, 1}()
        qr = QuadratureRule{RefQuadrilateral}(2)
    end
    cellvalues = CellValues(qr, ip);
    @timeit "create_dh" dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);
    
    Ndofs = ndofs(dh)
    
    K = allocate_matrix(dh)
    ch = ConstraintHandler(dh);
    
    ∂Ω = union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right"),
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
    );
    
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
    add!(ch, dbc);
    
    if dim == 3
        BB = Vec((1/sqrt(3),-1/sqrt(3),1/sqrt(3)))
    else
        BB = Vec((1/sqrt(2),-1/sqrt(2)))
    end
    
    close!(ch)
    
    function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
        n_basefuncs = getnbasefunctions(cellvalues)
        ## Reset to 0
        fill!(Ke, 0)
        fill!(fe, 0)
        ## Loop over quadrature points
        for q_point in 1:getnquadpoints(cellvalues)
            ## Get the quadrature weight
            dΩ = getdetJdV(cellvalues, q_point)
            ## Loop over test shape functions
            for i in 1:n_basefuncs
                δu  = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                ## Add contribution to fe
                fe[i] += δu * dΩ
                ## Loop over trial shape functions
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    ## Add contribution to Ke
                    Ke[i, j] += (∇δu ⋅ ∇u + BB ⋅ ∇u * δu) * dΩ
                end
            end
        end
        return Ke, fe
    end
    #md nothing # hide
    
    function assemble_global(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler)
        ## Allocate the element stiffness matrix and element force vector
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        fe = zeros(n_basefuncs)
        ## Allocate global force vector f
        f = zeros(ndofs(dh))
        ## Create an assembler
        assembler = start_assemble(K, f)
        ## Loop over all cels
        for cell in CellIterator(dh)
            ## Reinitialize cellvalues for this cell
            reinit!(cellvalues, cell)
            ## Compute element contribution
            assemble_element!(Ke, fe, cellvalues)
            ## Assemble Ke and fe into K and f
            assemble!(assembler, celldofs(cell), Ke, fe)
        end
        return K, f
    end
    #md nothing # hide
    
    K, f = assemble_global(cellvalues, K, dh);
    
    # Build analytical rhs for error
    ones = similar(f)
    ones .= 1.0
    apply!(K, ones, ch)
    
    f = K * ones
    apply!(K, f, ch)
    return K, f
end
