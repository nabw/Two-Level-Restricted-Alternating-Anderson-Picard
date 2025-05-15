using AAP, LinearAlgebra, CSV, PetscCall, JLD2

problems = (:plaplace, :oseen, :bidomain, :navierstokes)
cache_folder = ".aap-ran-cache"
mkpath("results")
mkpath(cache_folder)

parameters = Dict(:N => Dict(:plaplace => [8,16,32,64,128],
                             :oseen => [4,8,16,32,48],
                             :bidomain => [16,32,64,128,196],
                             :navierstokes => [32,64,128,256,384]),
                  :masks => Dict(:plaplace => [:none],
                                 :oseen => [:none,:u,:p],
                                 :bidomain => [:none,:e,:i],
                                 :navierstokes => [:none,:u,:p]),
                  :P => Dict(:plaplace => [1,2,3,4],
                             :oseen => [1,2,3,4],
                             :bidomain => [1,2,3],
                             :navierstokes => [1,2,3,4])
                )

adapts = (:none, :subselect_constant, :subselect_power, :random_constant, :random_power)

# Hard-code problems that don't converge to avoid errors
function nonConvergent(problem, adapt, mask, p, N)
    if problem == :navierstokes && mask == :p && adapt == :subselect_constant && p == 4 && N == 256
       return true
    elseif problem == :navierstokes && mask == :p && adapt == :subselect_constant && p == 3 && N == 384
       return true
    elseif problem == :navierstokes && mask == :p && adapt == :subselect_constant && p == 4 && N == 384
       return true
    elseif problem == :navierstokes && mask == :p && adapt == :subselect_power && p == 3 && N == 384
       return true
    elseif problem == :navierstokes && mask == :p && adapt == :subselect_power && p == 4 && N == 384
       return true
    elseif problem == :navierstokes && mask == :p && adapt == :random_power && p == 4 && N == 384
        return true
    elseif problem == :bidomain && mask == :none && adapt == :random_constant && p == 3 && N == 64
       return true
    elseif problem == :bidomain && mask == :i && adapt == :subselect_constant && p == 3 && N == 64
       return true
    elseif problem == :bidomain && mask == :e && adapt == :subselect_constant && p == 3 && N == 64
       return true
    elseif problem == :bidomain && mask == :e && adapt == :subselect_power && p == 3 && N == 64
       return true
    elseif problem == :bidomain && mask == :none && adapt == :subselect_constant && p == 3 && N == 128
       return true
    elseif problem == :bidomain && mask == :none && adapt == :subselect_power && p == 3 && N == 128
       return true
    elseif problem == :bidomain && mask == :none && adapt == :subselect_power && p == 3 && N == 128
       return true
    elseif problem == :bidomain && mask == :i && adapt == :random_power && p == 3 && N == 128
       return true
    elseif problem == :bidomain && mask == :e && adapt == :subselect_constant && p == 3 && N == 128
       return true
    else
       return false
    end
end

function generateArguments(problem, adapt, P)
    args = getArguments()
    args[:M] = 10
    args[:P] = P
    args[:maxit] = 500
    if problem == :navierstokes
        args[:reltol] = 1e-8
    elseif problem == :bidomain
        args[:M] = 50
        args[:reltol] = 1e-4
    else
        args[:reltol] = 1e-6
    end
    args[:adapt] = adapt
    return args
end

function getProblem(problem, N)
    @assert problem in problems "Typo in problem name"
    if problem == :plaplace
        p = 1.5
        β = 1e1
        ff, u, prec = ResidualPLaplace(N; p=p, β=β)
        return ff, u, Dict(:none => nothing), prec
    elseif problem == :oseen
        K, Mp, f, dofs_u, dofs_p = assembleOseen(N; μ=1.0)
        options = "-ksp_type preonly 
                    -pc_hypre_boomeramg_interp_type ext+i
                    -pc_hypre_boomeramg_coarsen_type HMIS
                    -pc_type hypre"
        PetscCall.init(args=split(options))

        Ndofs = length(f)
        u = zeros(Ndofs)
        prec = PetscCallPreconditioner(u, Mp, f)
        ff = linearResidual(K, prec, f)
        # We keep prec to destroy and avoid memory leak
        masks = Dict(:none => nothing,:u => dofs_u,:p =>dofs_p)
        return ff, u, masks, prec 

    elseif problem == :bidomain
        ff, u, dofs_e, dofs_i = ResidualEP(N)
        masks = Dict(:none=>nothing,:e=>dofs_e,:i=>dofs_i)
        return ff, u, masks
    elseif problem == :navierstokes
        Re = 5000.0
        ff, u, dofs_u, dofs_p = ResidualNS(N,μ=1.0/Re)
        masks = Dict(:none=>nothing,:u=>dofs_u,:p=>dofs_p)
        return ff, u, masks 
    end
end

function solveProblem(f,u,mask,problem,args)

    mode = problem == :navierstokes ? :picard : :residual
    # Problem specific arguments
    out = solveNonlinear(f, u, args;mask=mask,mode=mode)
    Ndofs, its, sol_time, errors, reorthogonalizations, adaptions = out
    return Ndofs, its, sol_time, errors, reorthogonalizations, adaptions
end

function solveAll(problems, parameters; warmup=false)
    for problem in problems
        suffix = warmup ? "  (warm up phase)" : ""
        println("==== Solving $(problem)"*suffix)
        Ns = parameters[:N][problem]
        Ps = parameters[:P][problem]
        if warmup
            Ns = [minimum(Ns)]
            Ps = [minimum(Ps)]
        end
        masks = parameters[:masks][problem]
        data_aap = []
        for N in Ns
            prob = getProblem(problem, N)
            ff, u, masks_pr = prob[1], prob[2], prob[3]
            u0 = copy(u)
            prec = length(prob) == 4 ? prob[4] : nothing
            Ndofs = length(u)
            for mask in masks
                for adapt in adapts
                    for P in Ps
                        # If cache file exists, then skip test as it was already ran
                        filename = "$(problem)-$(N)-$(mask)-$(adapt)-$(P).jld2"
                        its = sol_time = M = 0
                        dofs = Ndofs
                        if isfile("$(cache_folder)/$(filename)") && !warmup
                            dat = load("$(cache_folder)/$(filename)")
                            dofs = dat["dofs"]
                            its = dat["its"]
                            sol_time = dat["time"]
                            M = dat["M"]
                        else # No cache file or warmup
                            println("     dofs=$(Ndofs), mask=$(mask), adapt=$(adapt), P=$(P)")
                            if nonConvergent(problem, adapt, mask, P, N) # Skip if it gives errors
                                its = 0
                                sol_time = 0.0
                            else # Problem is registered to converge
                                args = generateArguments(problem,adapt,P)
                                u .= u0
                                mask_use = masks_pr[mask]
                                out = solveProblem(ff, u, mask_use,problem,args)
                                dofs, its, sol_time, errors, reorthogonalizations, adaptions = out
                                M = args[:M]
                                if !warmup && N == maximum(Ns) # Store residual evolution of largest problem
                                    data = (it=1:its, residuals=errors, reorthos=reorthogonalizations, adapts=adaptions, time=sol_time * ones(its))
                                    CSV.write("results/$(problem)-aap-$(P)-$(mask)-$(adapt).csv", data)
                                end
                                if !warmup # If not warmup, save cache file
                                    dat = Dict("dofs" => Ndofs, "its" => its, "time" => sol_time,
                                               "M" => M, "P" => P, "mask" => mask, "adapt" => adapt)
                                    save("$(cache_folder)/$(filename)", dat)
                                end # if !warmup
                            end # nonConvergent
                        end # isfile cache
                        push!(data_aap, (Ndofs, M, P, mask, adapt, its, sol_time))
                    end # Ps
                end # adapts
            end # masks
            !isnothing(prec) && destroy!(prec)
        end # Ns
        if !warmup
            dof_data = [d[1] for  d in data_aap]
            M_data = [d[2] for  d in data_aap]
            P_data = [d[3] for  d in data_aap]
            mask_data = [d[4] for  d in data_aap]
            adapt_data = [d[5] for  d in data_aap]
            its_data = [d[6] for  d in data_aap]
            time_data = [d[7] for  d in data_aap]
            data_out = (dofs=dof_data,
                    depth=M_data,
                    alt=P_data,
                    mask=mask_data,
                    adapt=adapt_data,
                    its=its_data,
                    time=time_data)
            CSV.write("results/$(problem)-aap.csv", data_out)
        end
    end
end

solveAll(problems, parameters; warmup=true) # JIT for all cases for better measurements
solveAll(problems, parameters; warmup=false)
