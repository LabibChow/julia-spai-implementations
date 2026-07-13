using MatrixMarket
using SparseArrays
using LinearAlgebra
using Plots

include(joinpath(@__DIR__, "..", "..", "Utils", "check_spd.jl"))
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "SPAI Iterations", "my_mr_spai.jl"))

# MR-SPAI test on HB/nos1

A = mmread("Matrices/tri100eigs4k.mtx")
A = sparse(Float64.(A))
n = size(A, 1)

println("=====")
println("MR-SPAI TEST")
println("Matrix: tri100eigs4k")
println("Size: ", size(A))
println("Nonzeros in A: ", nnz(A))
println("=====")

# 1. Check original matrix A

println("\nChecking whether original matrix A is SPD...")

spd_A = check_spd(A; name = "tri100eigs4k")

if !spd_A.is_spd
    error("tri100eigs4k is not SPD, stopping experiment.")
end

println("tri100eigs4k passed the SPD check.")

# 2. Build MR-SPAI

spai_itmax = 40
spai_tol = 1e-3

M0 = spzeros(Float64, n, n)

println("\nBuilding MR-SPAI...")

M_mr, res_mr = my_mr(A, M0, spai_itmax, spai_tol)

println("\nMR-SPAI construction done.")
println("SPAI iterations: ", length(res_mr) - 1)
println("Nonzeros in M_mr: ", nnz(M_mr))

# 3. Compute relative SPAI quality

I_sparse = spdiagm(0 => ones(Float64, n))

spai_quality =
    norm(I_sparse - A * M_mr) / norm(I_sparse)

println("SPAI quality ||I - A*M||_F / ||I||_F: ", spai_quality)

# 4. Check whether MR-SPAI is SPD

println("\nChecking whether MR-SPAI is SPD...")

spd_mr = check_spd(M_mr; name = "MR-SPAI Float64")

# 5. Plot MR-SPAI convergence

plot(
    0:length(res_mr)-1,
    res_mr,
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "MR-SPAI residual",
    title = "MR-SPAI convergence for tri100eigs4k",
    label = "MR Float64",
    linewidth = 3
)

savefig("tri100eigs4k_mr_spai_convergence.png")

# 6. Summary

println("\n=====")
println("SUMMARY")
println("=====")

println("Matrix: tri100eigs4k")
println("Method: MR-SPAI")
println("SPAI iterations: ", length(res_mr) - 1)
println("SPAI quality ||I - A*M||_F / ||I||_F: ", spai_quality)
println("Symmetry error: ", spd_mr.symmetry_error)
println("Smallest eigenvalue: ", spd_mr.lambda_min)
println("Largest eigenvalue: ", spd_mr.lambda_max)
println("Cholesky succeeded: ", spd_mr.cholesky_success)
println("SPD: ", spd_mr.is_spd)
println("Nonzeros in M: ", nnz(M_mr))
println("Saved plot: tri100eigs4k_mr_spai_convergence.png")