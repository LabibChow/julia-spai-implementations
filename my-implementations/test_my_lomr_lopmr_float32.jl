push!(LOAD_PATH, ".")

include("my-implementations/my_lomr_float32.jl")
include("my-implementations/my_lopmr_float32.jl")

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

M_lomr, R_lomr = my_lomr_float32(A, copy(M0), 20, Float32(1e-6))
println("Float32 LOMR residuals = ", R_lomr)
println("LOMR final residual = ", norm(I - A * Float64.(M_lomr)))
println("Type of LOMR M entries = ", eltype(M_lomr))

M_lopmr, R_lopmr = my_lopmr_float32(A, Pr, copy(M0), 20, Float32(1e-6))
println("Float32 LOPMR residuals = ", R_lopmr)
println("LOPMR final residual = ", norm(I - A * Float64.(M_lopmr)))
println("Type of LOPMR M entries = ", eltype(M_lopmr))