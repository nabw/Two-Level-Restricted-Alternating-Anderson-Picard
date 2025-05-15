module AAP

import Base: iterate
import LinearAlgebra: norm, axpy!, axpby!, mul!, diagm, inv, pinv, svd, svdvals, qr, qr!, LowerTriangular, UpperTriangular, dot, normalize!
#import PROPACK: tsvdvals_irl
import StatsBase: sample
import LinearAlgebra: ldiv!
import Random
Random.seed!(1234) # Fix seed

using Printf, TimerOutputs, ArgParse, InvertedIndices
using PetscCall, IterativeSolvers, QRupdate, LinearAlgebra, Preconditioners, AlgebraicMultigrid
using Ferrite, Tensors, Gridap, CSV, SparseArrays, JLD2
using ArgParse

to = TimerOutput()

include("AAPTests.jl")
export aap, aap!, PetscCallPreconditioner, destroy!
export getArguments, solveLinear, solveNonlinear, linearResidual

# Get all problem references
include("poisson.jl"); export assemblePoisson
include("adr.jl"); export assembleADR
include("stokes.jl"); export assembleStokes
include("oseen.jl"); export assembleOseen
include("plaplace.jl"); export ResidualPLaplace
include("hyperelasticity.jl"); export ResidualHyperelasticity
include("ep.jl"); export ResidualEP
include("navier-stokes.jl"); export ResidualNS


mutable struct State
    ORTH::String  # "" or "O", used for output
    ITER_TYPE::String # R or A, used for output
end

mutable struct QRaap{matT}
    R::matT
    size::Int64
end

mutable struct matG{T}
    G::AbstractMatrix{T}
    first_idx::Int64
    width::Int64
end

function generateRandomMask(mask, percentage::Float64)
    @assert 0 < percentage <= 100 "Sketching percentage must be in (0,100]"
    N = length(mask)
    out = sample(mask, convert(Int,floor(percentage/ 100.0 * N)), replace=false)
    sort!(out)
    return out
end # generateRandomMask

function estimate_smallest_singular_value(R::AbstractMatrix{T}) where {T}

    R = UpperTriangular(R)
    RT = LowerTriangular(R')
    
    # In the literature  some people suggest random inits, but
    # I saw no difference, so I preferred something more 
    # reproducible. N Barnafi 02/2025.
    x = ones(size(R, 1))
    y = similar(x)
    for i in 1:5
        # Copied from QRupdate
        ldiv!(y, RT, x)
        ldiv!(x, R, y)
        normalize!(x)
    end
    u = R*x
    val = dot(u, u) # it is <Rx, Rx> = l <x,x> = l, as x is normalized
    return sqrt(val) # abs is just a safe guard
    # Truncated randomized value. Equivalently fast, no difference in performance.
    #s, _, _, _ = tsvdvals_irl(R, k=1)
    #return s[1]
end

# This functions leverages circular memory access for efficiency.
function appendColumn!(A::matG{T}, v::AbstractVector{T}, iteration::Int64, depth::Int64) where {T}
    idx = mod1(iteration, depth)
    if A.width == depth
        A.first_idx = mod1(A.first_idx+1, depth) # round round round
    else
        A.width = min(A.width+1, depth) # Increase size until maximum width
    end
    copyto!(view(A.G,:, idx),v) # copy vector
end

mutable struct AAPIterable{funT, matQR, T}
    x::AbstractVector{T}
    f::funT
    mode::Symbol
    qr::QRaap{matQR} # Dense, so we hard-code the type
    r::AbstractVector{T}
    rPi::Union{Nothing,AbstractVector{T}}
    g::AbstractVector{T}
    work::AbstractVector{T}
    work_depth::AbstractVector{T}
    work_depth2::AbstractVector{T}
    work_depth3::AbstractVector{T}
    work_depth4::AbstractVector{T}
    weights::AbstractVector{T}
    dr::AbstractVector{T}
    drPi::Union{Nothing,AbstractVector{T}}
    dg::AbstractVector{T}
    dx::AbstractVector{T}
    #G::AbstractMatrix{T}
    G::matG{T}
    FPi::AbstractMatrix{T}
    atol::Float64
    rtol::Float64
    divtol::Float64
    residual::Float64
    res0::Float64
    dx_norm::Float64
    maxiter::Int64
    depth::Int64
    p::Int64
    Lip::Float64
    omega::Float64
    mask # ::Union{Vector{Int64}, Nothing}
    mask_adaptive # ::Union{Vector{Int64}, Nothing}
    qr_tol::Float64
    qr_max_it::Int
    qr_update::Symbol
    σ::Float64
    adaptive::Symbol
    state::State
    sketching_percentage::Float64
    reorthogonalizations::Int
    adapt_flag::Bool
end

# Required for 'iterate' interface.
@inline converged(it::AAPIterable) = it.residual ≤ it.atol || it.residual ≤ it.rtol * it.res0 || it.residual ≥ it.divtol || isnan(it.residual)
@inline start(it::AAPIterable) = 0
@inline done(it::AAPIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)

# Compute residual r = f(x) and its norm
function computeResidual!(it::AAPIterable{funT, matQR, T})  where {funT, matQR, T}

    if it.mode == :residual
        @timeit to "Copy" copy!(it.g, it.x) # start computing g
        @timeit to "Function eval" it.f(it.r, it.x)
        @timeit to "axpy" axpy!(it.omega, it.r, it.g) # g = x + f
    else # it.mode == :picard
        @timeit to "reset vector" it.r .= 0.0
        @timeit to "axpy" axpy!(-1.0/it.omega, it.x, it.r) # start computing r
        @timeit to "Function eval" it.f(it.g, it.x, it.omega)
        @timeit to "axpy" axpy!(1.0/it.omega, it.g, it.r) # g = x + ω f -> f=1/ω (g - x)
    end
    it.residual = norm(it.r)
end

function computeSketchingError(it, thresh; verbose=false)
    F = it.F
    n,m = size(F)
    Pi = zeros(n, n)
    Iss = zeros(n,n)  # S * pinv(S)
    for i in 1:size(it.mask,1)
        Pi[i,i] = it.mask[i]
        Iss[i,i] = 1.0 # localize 1 on non-zero singular vals
    end # i in size(it.mask,1)
    ## Computation with full svd
    #Im = diagm(ones(n))
    #mats = svd(F; full=true)
    #U = mats.U
    #S = mats.S
    #V = mats.V
    #matS = zeros(size(F))
    #matS[1:m,1:m] = diagm(S)
    #PiU = Pi * U
    #C = PiU' * PiU - Im
    #invS = pinv(matS)
    #Css = Iss * C * Iss
    #Yexact = Iss - Iss * inv(inv(C) + Iss) * Iss
    #Yapprx = Iss - Css + Css * Css
    #M1 = invS * (U' - Yexact * U' * Pi' ) * it.r
    #M2 = invS * (U' - Yapprx * U' * Pi' ) * it.r

    ## Computation with less algebra
    #mats = svd(F; full=true)
    #U = mats.U
    #S = mats.S
    #Vt = mats.Vt
    #matS = zeros(size(F))
    #matS[1:m,1:m] = diagm(S)
    #matS = diagm(S)
    current_size = 1:it.qr.size
    F = view(it.F,:,current_size)
    ex = qr(F) \ it.r
    M1 = ex - it.weights[current_size]
    norm_ex = norm(ex)
    n1 = norm(M1) / norm_ex # previous ones
    vals = svdvals(F)
    vals = vals[vals.>0.0]
    vals_Pi = svdvals(view(it.F, it.mask, current_size))
    vals_Pi = vals_Pi[vals_Pi.>0.0]

    cond = vals[1]/vals[end]
    cond_Pi = vals_Pi[1] / vals_Pi[end]
    error_good = n1 < thresh # less than 5% error
    
    verbose && println("\tSketching error=$(n1), cond=$(vals[1]/vals[end]), cond_Pi=$(vals_Pi[1]/vals_Pi[end])")
    #return cond > cond_Pi && error_good ? it.weights : ex
    return error_good ? it.weights[current_size] : ex
    #return it.weights[current_size]
end


function getMaskedPiVec(it::AAPIterable{funT, matQR, T}, X::AbstractVector{T}) where {funT, matQR, T}
    if !it.adapt_flag
        return X
    else
        return view(X, it.mask_adaptive)
    end
end

# override flag exists because we also use this function to update the rPi and drPi vectors, so that
# returning a reference to the smaller vectors does never update them. Leaving this comment after 
# a struggle much longer than what I deserved.
function getMaskedVec(it::AAPIterable{funT, matQR, T}, X::AbstractVector{T}, XPi::Union{Nothing,AbstractVector{T}}, override::Bool=false) where {funT, matQR, T}
    if isnothing(it.mask)
        out = X
    else
        out = override ? view(X, it.mask) : XPi
    end
    return getMaskedPiVec(it, out)
end

function getPi1MaskedVec(it::AAPIterable{funT, matQR, T}, X::AbstractVector{T}, XPi::Union{Nothing,AbstractVector{T}}, override::Bool=false) where {funT, matQR, T}
    if isnothing(it.mask)
        out = X
    else
        out = override ? view(X, it.mask) : XPi
    end
    return out
end


function getMaskedPiMat(it::AAPIterable{funT, matQR, T}, X::AbstractMatrix{T}) where {funT, matQR, T}
    if !it.adapt_flag  # If not adaptive or adaptivity not started
        return X
    else
        return view(X, it.mask_adaptive, :)
    end
end


function andersonStep!(it::AAPIterable{funT, matQR, T}) where {funT, matQR, T}

    # Anderson step, weights are updated here.
    if !it.adapt_flag && it.qr_update == :efficient
        r = getMaskedVec(it, it.r, it.rPi)
        FPi = getMaskedPiMat(it, it.FPi)
        work = getMaskedPiVec(it, it.work) # work is allocated in the Pi range
        @timeit to "csne solve" qr_its = csne!(it.qr.R, FPi, r, 
                                  it.weights, it.work_depth, 
                                  it.work_depth2, it.work_depth3, 
                                  work, it.qr.size; log=true, verbose=false,
                                  ortho_tol=it.qr_tol, ortho_max_it=it.qr_max_it)
        #it.weights[1:it.qr.size] .= computeSketchingError(it,0.5;verbose=true)
        it.reorthogonalizations += qr_its
    else # adaptive active or classic QR
        # Note: This is important because the ultra-efficient
        # updated QR is NOT the QR of the masked system.
        @timeit to "qr solve" begin
            act_cols = 1:it.qr.size
            FPi = view(getMaskedPiMat(it, it.FPi), :, act_cols)
            r = getMaskedVec(it, it.r, it.rPi)
            it.weights .= 0.0 # reset due to view upddate

            # After some tests, this is (surprisingly) faster
            # than preallocating or using views.
            tempF = copy(FPi)
            fact = qr!(tempF)

            if it.adaptive != :none 
                it.σ = estimate_smallest_singular_value(fact.R)
            end
            view(it.weights,act_cols) .= fact \ r
        end # timeit to qr solve
    end # adaptive not active

    # Now we do x <- x + r - (X + F) weights.
    # For loops looks weird due to circulant update of G.
    @timeit  to "Anderson update" begin
        axpy!(it.omega, it.r, it.x)
        for i in 1:it.G.width
            idx = mod1(it.G.first_idx + i - 1, it.depth)
            axpy!(-it.weights[i], view(it.G.G, :, idx), it.x)
        end
    end #timeit "Anderson update"
end # andersonStep

function deleteColumn(it::AAPIterable{funT, matQR, T}, step_type::Symbol) where {funT, matQR, T}

    if it.qr_update == :classic
        k_del = 1
        n = size(it.FPi,2)
        @inbounds for j in (k_del+1):n
            @views it.FPi[:,j-1] .= it.FPi[:, j]
        end # inbounds for
        it.FPi[:,n] .= zero(T)
        it.qr.size -= 1
    else # qr_update == :efficient
        # In efficient implementation, only remove column
        # on Anderson step.
        if step_type == :anderson
            qrdelcol!(it.FPi, it.qr.R, 1)
            it.qr.size -= 1
        end # step_type anderson
    end # it.qr_update classic
end

function addColumn(it::AAPIterable{funT, matQR, T}, step_type::Symbol) where {funT, matQR, T}
    # Note: Do full update always, truncate only for LS solution
    dr = getPi1MaskedVec(it, it.dr, it.drPi)
    FPi = it.FPi
    work = it.work
    if it.qr_update == :classic
        view(FPi,:,it.qr.size+1) .= dr
        it.qr.size += 1
    else # qr_update == :efficient
        # In efficient implementation, only add column 
        # on Anderson step.
        if step_type == :anderson
            qr_its = qraddcol!(FPi, it.qr.R, dr, it.qr.size, 
                               it.work_depth, it.work_depth2, 
                               it.work_depth3, it.work_depth4, 
                               work; updateMat=true, log=true, verbose=false, 
                               ortho_tol=it.qr_tol, ortho_max_it=it.qr_max_it)
            it.reorthogonalizations += qr_its
            it.qr.size += 1
        end # step_type == anderson
    end # it.qr_update == classic
end

################
##### AAP ######
################

function iterate(it::AAPIterable{funT, matQR, T}, iteration::Int=start(it)) where {funT, matQR, T}

    # Init iteration variables
    mk = min(iteration, it.depth)
    it.reorthogonalizations = 0
    it.adapt_flag = false # Always assume we don't adapt.
    step_type = (iteration+1) % it.p != 0 || iteration == 0 ? :richardson : :anderson
    it.dx_norm = max(it.dx_norm, norm(it.dx)) 

    # Compute current residual
    @timeit to "Copy" copy!(it.dx, it.x) # keep previous: dr = r_new - r_old
    @timeit to "Copy" copy!(it.dr, it.r) # keep previous: dr = r_new - r_old
    @timeit to "Copy" copy!(it.dg, it.g) # keep previous: dg = g_new - g_old
    @timeit to "Compute residual" computeResidual!(it) # update it.r, it.g and it.residual (norm(r))
    @timeit to "axpby" axpby!(1.0, it.r, -1.0, it.dr) # dr = r - r_old
    @timeit to "axpby" axpby!(1.0, it.g, -1.0, it.dg) # dg = r - r_old

    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # Update mask if doing adaptivity after updating rPi.
    @timeit to "Adaptive step" begin

        # We use adaptivity only on Anderson steps
        if it.adaptive != :none && iteration > 2 && step_type == :anderson 

            it.mask_adaptive = 1:length(it.work) # Always reset mask
            σ = 0.0
            if it.qr_update == :efficient
                size_R = it.qr.size
                vR = view(it.qr.R, 1:size_R, 1:size_R)
                @timeit to "Get σ" σ = estimate_smallest_singular_value(vR)
            else # qr_update classic
                σ = it.σ
            end
            
            if it.adaptive in (:subselect_constant, :random_constant)
                η = 1.0
            else # :subselect_power or random_power
                η = (iteration + 1.0)^(-1.1)
            end
            it.Lip = max(it.Lip, norm(it.dr) / it.dx_norm) # Lipschitz const estimate
            Ndofs = length(it.work)

            ϵ = η*σ*Ndofs / (it.Lip * it.residual * it.dx_norm) - 1

            #print("\tDEBUG: Adaptive values: σ=$(σ) ϵ=$(ϵ) Lip=$(it.Lip) dx=$(it.dx_norm)")
            if ϵ > 0 # LHS consistency
                perc = it.sketching_percentage

                r = getPi1MaskedVec(it, it.r, it.rPi)
                dr = getPi1MaskedVec(it, it.dr, it.drPi)
                N = Int(floor(perc * length(r)))
                if it.adaptive in (:subselect_constant, :subselect_power)
                    @timeit to "Compute indices" indices = partialsortperm(abs.(r), 1:N, rev=true)
                    sort!(indices)
                else # adaptive == :random_constant or :random_power
                    @timeit to "Compute indices" indices = generateRandomMask(1:length(it.work), 100 * perc)
                end
                ϵf = norm(view(dr, Not(indices))) / it.residual # Error of complement
                #print("  ϵf=$(ϵf)   Ntol=$(N)")
                if 0 < ϵf < ϵ # RHS consistency
                    it.mask_adaptive = indices
                    it.adapt_flag = true
                end # if ϵf < ϵ
            end # if ϵ > 0
            #println("")
        end # if adaptive
    end # timeit Adaptive step

    # shortcircuit didnt work, maybe problem with timer.
    # If adaptive, Pi vectors don't make sense
    @timeit to "Update Pi vectors" begin
        if !isnothing(it.mask)
            it.rPi .= 0.0 # zero out unused
            it.drPi .= 0.0  # zero out unused
            @timeit to "GetMaskedVec" getMaskedPiVec(it, it.rPi) .= getMaskedVec(it, it.r, it.rPi, true)
            @timeit to "GetMaskedVec" getMaskedPiVec(it, it.drPi) .= getMaskedVec(it, it.dr, it.drPi, true)
        end
    end # timeit Update Pi vectors

    # Update QR factorization of F and X
    @timeit to "Update Mats" begin
        if iteration > 0
            # Delete columns only after matrix is full.
            # In efficient case, we increase threshold as we only modify
            # the QR on anderson steps.
            thresh = it.qr_update == :classic ? it.depth : it.p * it.depth
            if iteration > thresh
                @timeit to "Delete column from F|R" deleteColumn(it, step_type)
            end # iteration > depth


            # When using efficient qr, we update matrices only on Anderson step
            if !(step_type == :richardson && it.qr_update == :efficient)
                idx = 0
                if it.qr_update == :classic
                    idx = iteration
                else # qr_update == anderson and step_type == anderson
                    #idx = mod1(iteration, depth)
                    idx = Int(floor((iteration-1)/it.p)) + 1
                end
                @timeit to "Add column to G" appendColumn!(it.G, it.dg, idx, it.depth)
            end

            @timeit to "Add column to F|R" addColumn(it, step_type)
        end # if iteration > 0
    end #timeit Update Mats

    # Iterate for new x
    if step_type == :richardson # (iteration+1) % it.p != 0 || iteration == 0
        @timeit to "Richardson" begin
            axpy!(it.omega, it.r, it.x)
            it.state.ITER_TYPE = "R"
        end #timeit Richardson
    else  # step_type == :anderson
        @timeit to "Anderson" begin
            andersonStep!(it)
            it.state.ITER_TYPE = "A"
        end #timeit Anderson
    end
    @timeit to "axpby" axpby!(1.0, it.x, -1.0, it.dx) # dx = x - x_old

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

# Utility functions

function allocate_work(x::AbstractVector{T}, mask::maskT) where {T, maskT}
    if isnothing(mask)
        return similar(x)
    else
        return zeros(length(mask))
    end
end

function aap_iterator(x, f;
                      mode::Symbol = :residual,
                      abstol::Float64 = zero(real(eltype(b))),
                      reltol::Float64 = sqrt(eps(real(eltype(b)))),
                      divtol::Float64 = 1e12,
                      maxiter::Int64 = size(A, 2),
                      depth::Int64=1,
                      p::Int64=1,
                      omega::Float64=1.0,
                      mask=nothing,
                      adaptive::Symbol=:none,
                      sketching_percentage::Float64=0.1,
                      qr_tol::Float64=1e-14,
                      qr_max_it::Int=1,
                      qr_update::Symbol=:classic # or efficient
                      )
    
    if mode == :residual
        r = similar(x)
        f(r, x) # In-place evaluation
        g = x+omega*r
    elseif mode == :picard
        g = similar(x)
        f(g, x, omega)
        r = 1/omega * (g-x)
    else
        throw("Execution mode must be either :residual or :picard, which
              is related to the fixed point iteration g(x) = x + f(x). More specifically,
              if an 'f' function is provided, use :residual. If a 'g' function is 
              provided, use :picard.")
    end
    
    res_norm = norm(r)
    atol = abstol
    rtol = reltol
    copyto!(x,g) # x1 = g0
    dr = similar(r)
    dg = similar(r)
    dx = similar(r)
    dx .= 0.0 # this avoids weird initial norm

    T = eltype(x)
    R = zeros(T, depth,depth)
    qr = QRaap(R, 0)

    @timeit to "Allocate work" work = allocate_work(x, mask)
    weights = zeros(T, depth)

    # For adaptivity we consider another mask. This avoids difficulties in handling
    # the matrix F, which is actually Pi⋅F by deafult, i.e. the initial mask times F.
    # This separation of masks is what allows us, in the case of having an initial mask,
    # to allocate F only in the masked dofs, so that the second mask acts on the masked object.
    # This also makes the handling of indexes easier. 
    mask_adaptive = nothing
    if adaptive != :none
        # We use vector to allow for in-place removal 
        # of dofs... thus the 'collect'.
        mask_adaptive = 1:length(work)
    end


    # Depth-sized (small) work vectors
    work_depth = zeros(T, depth)
    work_depth2 = similar(work_depth)
    work_depth3 = similar(work_depth)
    work_depth4 = similar(work_depth)
    
    # Allocate matrices
    @timeit to "Allocate Mat" G = zeros(T, size(x,1),depth)
    mG = matG(G, 1, 0)
    @timeit to "Allocate Mat" FPi = zeros(T, length(work), depth)

    # 'work' has the size of the reduced system, so it is used as a reference
    rPi = drPi = nothing
    if !isnothing(mask)
        rPi = similar(work)
        drPi = similar(work)
    end
    res0 = res_norm
    state = State("", "R")
    dx_norm = 0.0
    Lip = 0.0 # Initiali Lispchitz const
    reorthogonalizations=0
    adapt_flag=false
    σ=0.0

    AAPIterable(x, f, mode, qr, r, rPi, g, work, work_depth, work_depth2, work_depth3, work_depth4, weights, dr, drPi, dg, dx, mG, FPi, atol, rtol, divtol, res_norm, res0, dx_norm, maxiter, depth, p, Lip, omega, mask, mask_adaptive, qr_tol, qr_max_it, qr_update, σ, adaptive, state, sketching_percentage, reorthogonalizations, adapt_flag)
end

"""
    aap(A, b; kwargs...) -> x, [history]

Same as [`aap!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
aap(f; kwargs...) = aap!(zero(f(x)), g; kwargs...)

"""
    aap!(x, A, b; kwargs...) -> x, [history]

# Arguments

- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `Pl = Identity()`: left preconditioner of the method. Should be symmetric,
  positive-definite like `A`;
- `abstol::Real = zero(real(eltype(b)))`,
  `reltol::Real = sqrt(eps(real(eltype(b))))`: absolute and relative
  tolerance for the stopping condition
  `|r_k| ≤ max(reltol * |r_0|, abstol)`, where `r_k ≈ A * x_k - b`
  is approximately the residual in the `k`th iteration.
  !!! note
      The true residual norm is never explicitly computed during the iterations
      for performance reasons; it may accumulate rounding errors.
- `maxiter::Int = size(A,2)`: maximum number of iterations;
- `verbose::Bool = false`: print method information;
- `log::Bool = false`: keep track of the residual norm in each iteration.

# Output

**if `log` is `false`**

- `x`: approximated solution.

**if `log` is `true`**

- `x`: approximated solution.
- `ch`: convergence history.

**ConvergenceHistory keys**

- `:tol` => `::Real`: stopping tolerance.
- `:resnom` => `::Vector`: residual norm at each iteration.
"""
function aap!(x, f;
             mode::Symbol = :residual,
             abstol::Real = zero(real(eltype(x))),
             reltol::Real = sqrt(eps(real(eltype(x)))),
             maxiter::Int = length(x),
             log::Bool = false,
             verbose::Bool = false,
             depth::Int = 10,
             p::Int = 5,
             omega::Float64 = 1.0,
             mask=nothing,
             qr_tol::Float64 = 1e-14,
             qr_max_it::Int=2,
             qr_update=:classic,
             sketching_percentage::Float64=0.1,
             adaptive=:none)
    #@timeit "All" begin
    verbose && @printf("=== aap ===\n%4s\t%4s\t%7s\n","rest","iter","resnorm")

    if !(adaptive in (:none, :subselect_constant, :subselect_power, :random_constant, :random_power))
        println("Adaptive type not recognized, defaulting to none")
        adaptive = :none
    end

    # Actually perform AAP
    #@timeit "create iter" begin
    @timeit to "Create iterable" iterable = aap_iterator(x, f; mode = mode, abstol = abstol, reltol = reltol, maxiter = maxiter, depth = depth, p = p, omega = omega, mask=mask, adaptive=adaptive, sketching_percentage=sketching_percentage, qr_tol=qr_tol, qr_max_it=qr_max_it, qr_update=qr_update)
    if log
        errors = zeros(1)
        reorthogonalizations = zeros(1)
        adaptive_flags = zeros(1)
        errors[1] = iterable.residual
    end
    its = 0
    for (iteration, item) = enumerate(iterable)
        @timeit to "Solve" begin
            if log
                push!(errors, iterable.residual)
                push!(reorthogonalizations, iterable.reorthogonalizations)
                push!(adaptive_flags, iterable.adapt_flag)
            end
            verbose && @printf("%s %3d\te_abs=%1.2e\te_rel=%1.2e", iterable.state.ITER_TYPE, iteration, iterable.residual, iterable.residual / iterable.res0)
            if qr_max_it > 0 && verbose && qr_update == :efficient
                print("\treorthogonalizations=$(iterable.reorthogonalizations)")
            end
            if adaptive != :none && verbose
                print("\tadaptive active=$(iterable.adapt_flag)")
            end
            verbose && print("\n")
            its = iteration 
        end # timeit SOLVE
    end
    verbose && @printf("%s %3d\te_abs=%1.2e\te_rel=%1.2e\n", iterable.state.ITER_TYPE, its, iterable.residual, iterable.residual / iterable.res0)

    copy!(x, iterable.x)
    #end # timeit Final Pr

    verbose && println()


    if iterable.residual ≥ 1e12
        its = maxiter
    end
    if log
        return x, its, errors, reorthogonalizations, adaptive_flags
    else
        return x, its
    end
    #end #timeit All
end
end # module AAP
