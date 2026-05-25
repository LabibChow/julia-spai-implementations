push!(LOAD_PATH, ".")

include("my-implementations/my_lomr.jl")
include("my-implementations/my_lopmr.jl")

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

M_lomr, R_lomr = my_lomr(A, copy(M0), 20, 1e-10)
println("LOMR residuals = ", R_lomr)
println("LOMR final residual = ", norm(I - A*M_lomr))

M_lopmr, R_lopmr = my_lopmr(A, Pr, copy(M0), 20, 1e-10)
println("LOPMR residuals = ", R_lopmr)
println("LOPMR final residual = ", norm(I - A*M_lopmr))