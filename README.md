Julia library for executing the tests in the work "Two-level Sketching Alternating Anderson Acceleration for Complex Physics Applications" by NA Barnafi and M Lupo Pasini.

## Installation
This is a Julia library, and so to use it:

- [Download Julia](https://julialang.org/downloads/)
- Clone the library
```
git clone https://github.com/nabw/Two-Level-Restricted-Alternating-Anderson-Picard.git
```
- Install the library 
```
cd Two-Level-Restricted-Alternating-Anderson-Picard
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
- Test the installation
```
julia --project=. tests/runtests.jl
```

That's it. If the tests gave no errors, you are ready to go.

## Algorithm interface

AAP is basically an algorithm that works for abstract problems written either in residual

$$ F(x) = 0$$

or fixed-point form

$$ G(x) = x . $$

To run a code using the residual form, let's consider an example scalar problem given by finding a vector $x$ such that for some matrix $A$ and vector $b$ is holds that

$$ b - Ax = 0 . $$

```
using AAP, LinearAlgebra
A = diagm([1.0,2.0,3.0])
b = [1.0,1.0,1.0]

function f(r, x)
    r .= b - A*x
end

x0 = rand(3)
out = AAP.aap!(x0, f; maxiter=10, p=2)
println("Solution: $(out[1])")
println("Iterations: $(out[2])")
```

The same can be writing the iteration as a fixed-point problem. 

```
using AAP, LinearAlgebra
A = diagm([1.0,2.0,3.0])
b = [1.0,1.0,1.0]

function f(g, x, w)
    g .= x + w * (b - A*x)
end

x0 = rand(3)
out = AAP.aap!(x0, f; mode=:picard, maxiter=10, p=2)
println("Solution: $(out[1])")
println("Iterations: $(out[2])")
```

These examples can be run interactively in the Julia REPL using `julia --project=.`. 

## Examples

All examples follow the same interface using `ArgParse`, and thus can be fully configured from `bash` (or equivalent). Running

```
> julia --project=. tests/poisson.jl -help
```
will print the following: 
```
usage: poisson.jl [-N N] [-M M] [-P P] [-S S] [--reltol RELTOL]
                  [--abstol ABSTOL] [--maxit MAXIT] [--adapt ADAPT]
                  [--qr-update QR-UPDATE] [--mask MASK]
                  [--solver SOLVER] [--verbose] [-h]

optional arguments:
  -N N                  Number of elements per side (type: Int64,
                        default: 10)
  -M M                  Anderson depth (type: Int64, default: 10)
  -P P                  Anderson alternation parameter (type: Int64,
                        default: 1)
  -S S                  Sketching percentage (type: Float64, default:
                        0.3)
  --reltol RELTOL       Absolute tolerance in AAP (type: Float64,
                        default: 1.0e-6)
  --abstol ABSTOL       Absolute tolerance in AAP (type: Float64,
                        default: 1.0e-10)
  --maxit MAXIT         Maximum AAP iterations (type: Int64, default:
                        1000)
  --adapt ADAPT         Adaptive type
                        none|subselect_power|subselect_const|random_power|random_const
                        (default: "none")
  --qr-update QR-UPDATE
                        Set QR update type classic|efficient (default:
                        "classic")
  --mask MASK           Set field for mask in block problems
                        u|p|e|i|none (default: "none")
  --solver SOLVER       Define solver type for linear problems:
                        aap|gmres (default: "aap")
  --verbose             Activate verbose flag in AAP
  -h, --help            show this help message and exit

```
Then to solve the Poisson problem in 3D with 30 elements per side using AAP(10,2) with subselect constant adaptivity of a 10% simply run

```
> julia --project=. tests/poisson.jl -N 100 -M 10 -P 2 -S 0.1 --adapt subselect_constant --verbose
```
The `--verbose` flag activates the output of the method. This yields the following:

```
=== aap ===
rest	iter	resnorm
R   1	e_abs=1.47e+01	e_rel=3.24e-01	adaptive active=false
A   2	e_abs=1.03e+01	e_rel=2.26e-01	adaptive active=false
R   3	e_abs=6.19e+00	e_rel=1.36e-01	adaptive active=false
A   4	e_abs=5.47e+00	e_rel=1.20e-01	adaptive active=true
R   5	e_abs=3.58e+00	e_rel=7.87e-02	adaptive active=false
A   6	e_abs=3.28e+00	e_rel=7.22e-02	adaptive active=true
R   7	e_abs=1.16e+01	e_rel=2.54e-01	adaptive active=false
A   8	e_abs=8.83e+00	e_rel=1.94e-01	adaptive active=true
R   9	e_abs=2.00e+00	e_rel=4.39e-02	adaptive active=false
A  10	e_abs=1.88e+00	e_rel=4.14e-02	adaptive active=false
R  11	e_abs=7.64e-02	e_rel=1.68e-03	adaptive active=false
A  12	e_abs=5.26e-02	e_rel=1.16e-03	adaptive active=false
R  13	e_abs=1.70e-02	e_rel=3.75e-04	adaptive active=false
A  14	e_abs=1.50e-02	e_rel=3.29e-04	adaptive active=true
R  15	e_abs=2.52e-02	e_rel=5.54e-04	adaptive active=false
A  16	e_abs=1.89e-02	e_rel=4.17e-04	adaptive active=true
R  17	e_abs=6.21e-02	e_rel=1.37e-03	adaptive active=false
A  18	e_abs=5.86e-02	e_rel=1.29e-03	adaptive active=true
R  19	e_abs=7.80e-03	e_rel=1.72e-04	adaptive active=false
A  20	e_abs=6.15e-03	e_rel=1.35e-04	adaptive active=true
R  21	e_abs=4.49e-03	e_rel=9.89e-05	adaptive active=false
A  22	e_abs=4.18e-03	e_rel=9.19e-05	adaptive active=true
R  23	e_abs=6.51e-03	e_rel=1.43e-04	adaptive active=false
A  24	e_abs=5.45e-03	e_rel=1.20e-04	adaptive active=true
R  25	e_abs=9.11e-04	e_rel=2.01e-05	adaptive active=false
A  26	e_abs=7.99e-04	e_rel=1.76e-05	adaptive active=true
R  27	e_abs=6.08e-04	e_rel=1.34e-05	adaptive active=false
A  28	e_abs=4.96e-04	e_rel=1.09e-05	adaptive active=true
R  29	e_abs=1.16e-04	e_rel=2.56e-06	adaptive active=false
A  30	e_abs=1.01e-04	e_rel=2.23e-06	adaptive active=true
R  31	e_abs=4.56e-05	e_rel=1.00e-06	adaptive active=false
R  31	e_abs=3.98e-05	e_rel=8.76e-07

Dofs:29791
Iterations:31
Solution time:4.774393081665039

```
The first column shows the type of iteration (**A**nderson or **R**ichardson/Picard). Then follow the iteration number, absolute and relative errors, and at the end we see whether adaptivity was used in that iteration. Better time measurements can be obtained by running the solution process twice _within the same script_, as Julia uses JIT (Just-In-Time) compilation, so the first run is commonly slower (and referred to as _warm-up_). If we add a warm-up run to the same Poisson example, we get an execeution time of 0.074 seconds, which is more reasonable for that problem size.  _Note: that some robustness to an increase of DoFs is seen as the residual used considers an Incomplete Cholesky preconditioner._

All tests were executed using the `tests/aap-tests/jl` script, whose results are reported in our paper. 
