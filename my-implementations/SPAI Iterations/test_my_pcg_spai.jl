push!(LOAD_PATH, ".")

include("my-implementations/my_pcg.jl")

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

M, R_norm = my_pcg(A, Pr, copy(M0), 20, 1e-10)

println(R_norm)
println("Final residual = ", norm(I - A*M))