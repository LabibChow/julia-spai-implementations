using MatrixDepot
using SparseArrays
using LinearAlgebra
using Plots

include(joinpath(@__DIR__, "..", "..", "Utils", "check_spd.jl"))
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_cg_solve.jl"))
include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_pcg_solve.jl"))

include(joinpath(@__DIR__, "..", "..", "SPAI Iterations", "my_lomr_spai.jl"))

# LOMR-SPAI test on HB/nos1

md = mdopen("HB/nos1")

A = sparse(Float64.(md.A))
n = size(A, 1)

println("=====")
println("LOMR-SPAI TEST")
println("Matrix: HB/nos1")
println("Size: ", size(A))
println("Nonzeros in A: ", nnz(A))
println("=====")

# 1. Check original matrix A

println("\nChecking whether original matrix A is SPD...")

spd_A = check_spd(A; name = "HB/nos1")

if !spd_A.is_spd
    error("HB/nos1 is not SPD, stopping experiment.")
end

println("HB/nos1 passed the SPD check.")

# 2. Build LOMR-SPAI

spai_itmax = 40
spai_tol = 1e-3

M0 = spzeros(Float64, n, n)

println("\nBuilding LOMR-SPAI...")

M_lomr, res_lomr = my_lomr(A, M0, spai_itmax, spai_tol)

println("\nLOMR-SPAI construction done.")
println("SPAI iterations: ", length(res_lomr) - 1)
println("Nonzeros in M_lomr: ", nnz(M_lomr))

# 3. Compute SPAI quality

I_sparse = spdiagm(0 => ones(Float64, n))

spai_quality =
    norm(I_sparse - A * M_lomr) / norm(I_sparse)

println("SPAI quality ||I - A*M||_F / ||I||_F: ", spai_quality)

# 4. Check whether LOMR-SPAI is SPD

println("\nChecking whether LOMR-SPAI is SPD...")

spd_lomr = check_spd(M_lomr; name = "LOMR-SPAI Float64")

# 5. Plot LOMR-SPAI convergence

plot(
    0:length(res_lomr)-1,
    res_lomr,
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "LOMR-SPAI residual",
    title = "LOMR-SPAI convergence for HB/nos1",
    label = "LOMR Float64",
    linewidth = 3
)

savefig("nos1_lomr_spai_convergence.png")

# 6. If LOMR-SPAI is SPD, run PCG

if spd_lomr.is_spd

    println("\nLOMR-SPAI passed SPD check. Running PCG...")

    x_exact = ones(Float64, n)
    b = A * x_exact
    x0 = zeros(Float64, n)

    solver_itmax = 1000
    solver_tol = 1e-5

    x_pcg_lomr, res_pcg_lomr =
        my_pcg_solve(A, b, x0, M_lomr, solver_itmax, solver_tol)

    pcg_lomr_rel_res =
        norm(b - A * x_pcg_lomr) / norm(b)

    pcg_lomr_rel_err =
        norm(x_pcg_lomr - x_exact) / norm(x_exact)

else
    println("\nLOMR-SPAI is not SPD. PCG will be skipped.")

    res_pcg_lomr = Float64[]
    pcg_lomr_rel_res = NaN
    pcg_lomr_rel_err = NaN
end

# 7. Summary

println("\n=====")
println("SUMMARY")
println("=====")

println("Matrix: HB/nos1")
println("Method: LOMR-SPAI")
println("SPAI iterations: ", length(res_lomr) - 1)
println("SPAI quality ||I - A*M||_F / ||I||_F: ", spai_quality)
println("Symmetry error: ", spd_lomr.symmetry_error)
println("Smallest eigenvalue: ", spd_lomr.lambda_min)
println("Largest eigenvalue: ", spd_lomr.lambda_max)
println("Cholesky succeeded: ", spd_lomr.cholesky_success)
println("SPD: ", spd_lomr.is_spd)
println("Nonzeros in M: ", nnz(M_lomr))
println("Saved plot: nos1_lomr_spai_convergence.png")

if spd_lomr.is_spd
    println("\nPCG with LOMR-SPAI")
    println("Iterations: ", length(res_pcg_lomr) - 1)
    println("Final residual: ", pcg_lomr_rel_res)
    println("Relative error: ", pcg_lomr_rel_err)
else
    println("\nPCG with LOMR-SPAI was skipped because M_lomr was not SPD.")
end