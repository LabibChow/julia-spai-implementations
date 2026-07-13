push!(LOAD_PATH, ".")

include(joinpath(@__DIR__, "my_pcg_float32_spai.jl"))

using SparseArrays
using LinearAlgebra

n = 5

A = spdiagm(
    0 => fill(2.0, n),
    -1 => fill(-1.0, n-1),
    1 => fill(-1.0, n-1)
)

M0 = spdiagm(0 => ones(n))

M, R_norm = my_cg_float32(A, copy(M0), 20, Float32(1e-6))

println("Float32 CG residuals = ", R_norm)
println("Final residual = ", norm(I - A * Float64.(M)))
println("Type of M entries = ", eltype(M))