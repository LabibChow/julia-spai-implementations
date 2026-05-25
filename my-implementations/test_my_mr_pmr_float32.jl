push!(LOAD_PATH, ".")

include("my-implementations/my_mr_float32.jl")
include("my-implementations/my_pmr_float32.jl")

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

M_mr, R_mr = my_mr_float32(A, copy(M0), 20, Float32(1e-6))
println("Float32 MR residuals = ", R_mr)
println("MR final residual = ", norm(I - A * Float64.(M_mr)))
println("Type of MR M entries = ", eltype(M_mr))

M_pmr, R_pmr = my_pmr_float32(A, Pr, copy(M0), 20, Float32(1e-6))
println("Float32 PMR residuals = ", R_pmr)
println("PMR final residual = ", norm(I - A * Float64.(M_pmr)))
println("Type of PMR M entries = ", eltype(M_pmr))