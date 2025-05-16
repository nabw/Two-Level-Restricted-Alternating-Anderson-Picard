Julia library for executing the tests in the work "Two-level Sketching Alternating Anderson Acceleration for Complex Physics Applications" by NA Barnafi and M Lupo Pasini.

## Installation
This is a Julia library, and so to use it:

- [Download Julia](https://julialang.org/downloads/). 
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

That's is. If the tests run Ok, you are ready to go.

## Algorithm interface

AAP is basically an algorithm that works for abstract problems written either in residual

$$ F(x) = 0$$

or fixed-point form

$$ G(x) = x . $$

To run a code using the residual form, let's consider an example scalar problem given by finding a vector $x$ such that for some matrix $A$ and vector $b$ is holds that

$$ b - Ax = 0 . $$

```
using AAP
using LinearAlgebra
A = diagm([1.0,2.0,3.0])
b = [1.0,1.0,1.0]

function f(r, x)
    r .= b - A*x
end

x0 = np.rand(3)
out = AAP.aap!(x0, f; maxiter=10, p=2)
println("Solution: $(out[1])")
println("Iterations: $(out[2])")
```

The same can be writing the iteration as a fixed-point problem. 

```
using AAP
using LinearAlgebra
A = diagm([1.0,2.0,3.0])
b = [1.0,1.0,1.0]

function f(g, x, w)
    g .= x + w * (b - A*x)
end

x0 = np.rand(3)
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
Then solving the Poisson problem in 3D with 30 elements per side using AAP(10,2) with subselect constant adaptivity of a 10% simply run

```
> julia --project=. tests/poisson.jl -N 100 -M 10 -P 2 -S 0.1 --adapt subselect_const --verbose
```
The `--verbose` flag activates the output of the method. 
