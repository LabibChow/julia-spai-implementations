using MatrixDepot
using SparseArrays
using LinearAlgebra

include("my_cg.jl")

md = mdopen("HB/nos1")
A = md.A

A = Float64.(A)

n = size(A, 1)
x_exact = ones(Float64, n)
b = A * x_exact
x0 = zeros(Float64, n)

tol = 1e-8
maxit = 1000

x, residuals = my_cg(A, b, x0, tol, maxit)

rel_res = norm(b - A * x) / norm(b)
rel_err = norm(x - x_exact) / norm(x_exact)

println("Matrix: HB/nos1")
println("Size: ", size(A))
println("Nonzeros: ", nnz(A))
println("Final relative residual: ", rel_res)
println("Relative error: ", rel_err)
println("Iterations: ", length(residuals) - 1)