using MatrixMarket
using SparseArrays
using LinearAlgebra

A = mmread("Matrices/tri100eigs4k.mtx")
A = sparse(Float64.(A))

println("Matrix: tri100eigs4k")
println("Size: ", size(A))
println("Nonzeros: ", nnz(A))
println("Element type: ", eltype(A))

symmetry_error = norm(A - A') / norm(A)

println("\nSymmetry check")
println("Symmetry error: ", symmetry_error)

cholesky_success = try
    cholesky(A)
    true
catch err
    println("\nCholesky failed:")
    println(err)
    false
end

println("\nSPD result")
println("Cholesky succeeded: ", cholesky_success)
println("Classified as SPD: ", cholesky_success && symmetry_error < 1e-12)