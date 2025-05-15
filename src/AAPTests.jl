
mutable struct PetscCallPreconditioner{A, B, C, D, E, T}
    setup::PetscCall.KspSeqSetup{A,B,C,D,E}
    x::AbstractVector{T}
    M::AbstractMatrix{T}
    b::AbstractVector{T}
    temp::AbstractVector{T}
end

function PetscCallPreconditioner(x::AbstractVector{T},
                                 M::AbstractMatrix{T},
                                 b::AbstractVector{T}) where {T}

    setup = PetscCall.ksp_setup(x, M, b)
    temp = similar(b)
    temp .= 0.0
    return PetscCallPreconditioner(setup, x, M, b, temp)
end

function ldiv!(P::PetscCallPreconditioner{A,B,C,D,E,T},
               x::AbstractVector{T}) where {A,B,C,D,E,T}
    P.temp .= 0.0
    PetscCall.ksp_solve!(P.temp, P.setup, x)
    x .= P.temp
end

function ldiv!(y::AbstractVector{T},
               P::PetscCallPreconditioner{A,B,C,D,E,T},
               x::AbstractVector{T}) where {A,B,C,D,E,T}
    y .= 0.0
    PetscCall.ksp_solve!(y, P.setup, x)
end

function destroy!(P::PetscCallPreconditioner{A,B,C,D,E,T}) where {A,B,C,D,E,T}
    PetscCall.ksp_finalize!(P.setup)
end

function getArguments()
    # Get arguments
    s = ArgParseSettings()
    @add_arg_table! s begin
        "-N"
            help= "Number of elements per side"
            default = 10
            arg_type = Int
        "-M"
            help="Anderson depth"
            default = 10
            arg_type = Int
        "-P"
            help="Anderson alternation parameter"
            default = 1
            arg_type = Int
        "-S"
            help="Sketching percentage"
            default = 0.3
            arg_type = Float64
        "--reltol"
            help="Absolute tolerance in AAP"
            default = 1e-6
            arg_type = Float64
        "--abstol"
            help="Absolute tolerance in AAP"
            default = 1e-10
            arg_type = Float64
        "--maxit"
            help="Maximum AAP iterations"
            default = 1000
            arg_type = Int
        "--adapt"
            help = "Adaptive type none|subselect_power|subselect_const|random_power|random_const"
            default = "none"
        "--qr-update"
            help = "Set QR update type classic|efficient"
            default = "classic"
        "--mask"
            help = "Set field for mask in block problems  u|p|e|i|none"
            default = "none"
        "--solver"
            help = "Define solver type for linear problems: aap|gmres"
            default = "aap"
        "--verbose"
            help = "Activate verbose flag in AAP"
            action = :store_true
    end
    # Get args test
    out = Dict( :N => parse_args(s)["N"],
            :M => parse_args(s)["M"],
            :P => parse_args(s)["P"],
            :S => parse_args(s)["S"],
            :adapt => Symbol(parse_args(s)["adapt"]),
            :qrupdate => Symbol(parse_args(s)["qr-update"]),
            :solver => Symbol(parse_args(s)["solver"]),
            :mask => Symbol(parse_args(s)["mask"]),
            :verbose => parse_args(s)["verbose"],
            :abstol => parse_args(s)["abstol"],
            :reltol => parse_args(s)["reltol"],
            :maxit => parse_args(s)["maxit"]
            )
    return out
end

function linearResidual(K, Prec, f)
    function g(r, x)
        mul!(r, K, x) # xwork = K xin
        axpby!(1, f, -1, r) # work = f - work = f - K xin
        ldiv!(Prec, r) # work = P ( residual )
    end
    return g
end

function solveLinear(K, f, u, Prec, args; mask=nothing)

    M = args[:M]
    P = args[:P]
    S = args[:S]
    ADAPTIVE = args[:adapt]
    UPDATE = args[:qrupdate]
    SOLVER = args[:solver]
    verbose = args[:verbose]
    reltol = args[:reltol]
    abstol = args[:abstol]
    maxit = args[:maxit]
    Ndofs = length(u)
    if SOLVER == :aap
        g = linearResidual(K, Prec, f)
        t0 = time()
        x, its, errors, reorthos, adapts = AAP.aap!(u, g; log=true, maxiter=maxit, verbose=verbose, abstol=abstol, reltol=reltol, depth=M, p=P, adaptive=ADAPTIVE, qr_max_it=2, qr_update=UPDATE, mask=mask, sketching_percentage=S)
        sol_time = time() - t0
        return Ndofs, its, sol_time, errors, reorthos, adapts
    else # == "gmres"
        x, ch_cg = IterativeSolvers.gmres!(u, K, f, Pl=Prec; log=true, maxiter=maxit, verbose=verbose, abstol=abstol, reltol=reltol, restart=M)
        its = ch_cg.iters
    end
    return its
end

function solveNonlinear(f, u, args; mask=nothing, mode=:residual)
    Ndofs = length(u)
    M = args[:M]
    P = args[:P]
    S = args[:S]
    ADAPT = args[:adapt]
    ADAPTIVE = args[:adapt]
    UPDATE = args[:qrupdate]
    verbose = args[:verbose]
    reltol = args[:reltol]
    abstol = args[:abstol]
    maxit = args[:maxit]
    Ndofs = length(u)
    t0 = time()
    out = AAP.aap!(u, f; maxiter=maxit, abstol=abstol, reltol=reltol, verbose=verbose, p=P, depth=M, omega=1.0, adaptive=ADAPT, sketching_percentage=S, mask=mask, mode=mode, log=true)
    sol_time = time() - t0
    x, its, errors, reorthos, adapts = out
    return Ndofs, its, sol_time, errors, reorthos, adapts
end

