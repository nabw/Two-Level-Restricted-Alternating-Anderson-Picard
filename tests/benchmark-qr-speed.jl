import StatsBase: sample
using ProgressBars
using LinearAlgebra
using Formatting

# Get cases of interest
exp_min = 10
exp_max = 21
Ns = [Int(2^n) for n in exp_min:exp_max]

tol = 1e-5

function compute_percentage(t_full, A::AbstractMatrix{T}) where {T}
    N,M = size(A)
    mask = collect(1:N)
    step = Int(floor(N/100)) # 1% steps
    if step < 1
        step = 1
    end

    for p in step:step:(N-1)
        submask = sample(mask, p, replace=false)
        t0 = time()
        vA = view(A, submask, :)
        fact = qr(vA)
        t_mask = time() - t0
        if t_mask > t_full
            return p/N
        end
    end
    return 1.0
end

function get_full(A::AbstractMatrix{T}) where {T}
    N,M = size(A)
    t0 = time()
    fact = qr(A)
    t_full = time() - t0
    return t_full
end

avg(ls) = sum(ls) / length(ls)

function get_stationary(f_time, A::AbstractMatrix{T}, tol, min_its) where {T}


    sum_tot = f_time(A)
    avg_old = sum_tot
    N,M = size(A)
    it = 1
    while true
        pnew = f_time(A)
        sum_tot += pnew
        it += 1
        avg_new = sum_tot / it
        error = abs(avg_new - avg_old) / avg_new 
        print("\r\tStationary avg for $(N), avg=$(sprintf1("%6.3e",avg_new))  it $(sprintf1("%7.0f",it)) error=$(sprintf1("%6.2e",error))")
        if error < tol && it > min_its # min sample size
            print("\e[2K") # Clear line content
            print("\r")
            return avg_new
        end
        avg_old = avg_new
    end
end



function compute_times(Ns)
    percentages = []
    for N in Ns
        println("\rComputing times for $(N)")
        A = rand(N, 20);
    
        println("\t Compute full stationary: ")
        min_its = 100
        t_full = get_stationary(get_full, A, tol, min_its) # Define function to get percentages
        
        min_avg = 100
        get_avg(AA) = compute_percentage(t_full,AA)

        # First, time it takes to go through the array as it is
        println("\t Compute masked stationary: ")
        avg = get_stationary(get_avg, A, tol, min_avg)
        append!(percentages, avg)
    end # Ns
    return percentages
end

percentages = compute_times(Ns)
println("")
println("")
println("==== Results ====")
for i in eachindex(Ns)
    println("For \t$(Ns[i])\t fraction: $(percentages[i])")
end
