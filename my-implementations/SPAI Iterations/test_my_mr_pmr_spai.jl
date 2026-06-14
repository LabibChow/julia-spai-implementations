push!(LOAD_PATH, ".")

include("my-implementations/my_mr.jl")
include("my-implementations/my_pmr.jl")

using SparseArrays
using LinearAlgebra

n = 5

A = spdiagm(
    0 => fill(2.0, n),
    -1 => fill(-1.0, n-1),
    1 => fill(-1.0, n-1)
)

Pr = spdiagm(0 => diag(A).^-1)
M0 = spdiagm(0 => ones(n))

M_mr, R_mr = my_mr(A, copy(M0), 20, 1e-10)
println("MR residuals = ", R_mr)
println("MR final residual = ", norm(I - A*M_mr))

M_pmr, R_pmr = my_pmr(A, Pr, copy(M0), 20, 1e-10)
println("PMR residuals = ", R_pmr)
println("PMR final residual = ", norm(I - A*M_pmr))