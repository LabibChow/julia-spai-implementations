# using MatrixDepot
using MatrixMarket
using SparseArrays
using LinearAlgebra
using Plots

include(joinpath(@__DIR__, "..", "..", "Utils", "check_spd.jl"))
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_cg_solve.jl"))
include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_pcg_solve.jl"))

include(joinpath(@__DIR__, "..", "..", "SPAI Iterations", "my_mr_spai.jl"))

# PCG with valid MR-SPAI on HB/nos1

A = mmread("Matrices/tri100eigs4k.mtx")
A = sparse(Float64.(A))
n = size(A, 1)

println("======================================")
println("PCG WITH MR-SPAI")
println("Matrix: tri100eigs4k")
println("Size: ", size(A))
println("Nonzeros in A: ", nnz(A))
println("======================================")

# 1. Check whether A is SPD

println("\nChecking whether original matrix A is SPD...")

spd_A = check_spd(A; name = "tri100eigs4k")

if !spd_A.is_spd
    error("tri100eigs4k is not SPD, stopping experiment.")
end

# 2. Build MR-SPAI

spai_itmax = 40
spai_tol = 1e-3

M0 = spzeros(Float64, n, n)

println("\nBuilding MR-SPAI...")

M_mr, res_mr = my_mr(A, M0, spai_itmax, spai_tol)

I_sparse = spdiagm(0 => ones(Float64, n))

spai_quality =
    norm(I_sparse - A * M_mr) / norm(I_sparse)

println("\nMR-SPAI construction done.")
println("SPAI iterations: ", length(res_mr) - 1)
println("SPAI quality ||I - A*M||_F / ||I||_F: ", spai_quality)
println("Nonzeros in M_mr: ", nnz(M_mr))

# 3. Check whether MR-SPAI is SPD

println("\nChecking whether MR-SPAI is SPD...")

spd_mr = check_spd(M_mr; name = "MR-SPAI Float64")

if !spd_mr.is_spd
    error("MR-SPAI is not SPD, so PCG with MR-SPAI will not be run.")
end

println("MR-SPAI passed SPD check. Proceeding to PCG.")

# 4. Create Ax = b

x_exact = ones(Float64, n)
b = A * x_exact
x0 = zeros(Float64, n)

solver_itmax = 1000
solver_tol = 1e-5

# 5. Run normal CG

println("\nRunning CG without preconditioner...")

x_cg, res_cg =
    my_cg_solve(A, b, x0, solver_itmax, solver_tol)

cg_rel_res =
    norm(b - A * x_cg) / norm(b)

cg_rel_err =
    norm(x_cg - x_exact) / norm(x_exact)

# 6. Run PCG with identity

M_identity = spdiagm(0 => ones(Float64, n))

println("\nRunning PCG with identity preconditioner...")

x_pcg_id, res_pcg_id =
    my_pcg_solve(A, b, x0, M_identity, solver_itmax, solver_tol)

pcg_id_rel_res =
    norm(b - A * x_pcg_id) / norm(b)

pcg_id_rel_err =
    norm(x_pcg_id - x_exact) / norm(x_exact)

# 7. Run PCG with MR-SPAI

println("\nRunning PCG with MR-SPAI preconditioner...")

x_pcg_mr, res_pcg_mr =
    my_pcg_solve(A, b, x0, M_mr, solver_itmax, solver_tol)

pcg_mr_rel_res =
    norm(b - A * x_pcg_mr) / norm(b)

pcg_mr_rel_err =
    norm(x_pcg_mr - x_exact) / norm(x_exact)

# 8. Plot solver convergence

p = plot(
    0:length(res_cg)-1,
    res_cg,
    yscale = :log10,
    xlabel = "Solver iteration",
    ylabel = "Relative residual",
    title = "CG vs PCG with MR-SPAI for tri100eigs4k",
    label = "CG",
    linewidth = 3
)

plot!(
    p,
    0:length(res_pcg_id)-1,
    res_pcg_id,
    label = "PCG Identity",
    linestyle = :dash,
    linewidth = 2
)

plot!(
    p,
    0:length(res_pcg_mr)-1,
    res_pcg_mr,
    label = "PCG MR-SPAI",
    linestyle = :dot,
    linewidth = 3
)

savefig(p, "tri100eigs4k_cg_vs_pcg_mr_spai.png")

println("\nSaved plot: tri100eigs4k_cg_vs_pcg_mr_spai.png")

# 9. Summary

println("\n=====")
println("SUMMARY")
println("=====")

println("\nSPAI information")
println("MR-SPAI quality: ", spai_quality)
println("MR-SPAI smallest eigenvalue: ", spd_mr.lambda_min)
println("MR-SPAI Cholesky succeeded: ", spd_mr.cholesky_success)
println("MR-SPAI SPD: ", spd_mr.is_spd)
println("Nonzeros in M_mr: ", nnz(M_mr))

println("\nSolver iteration counts")
println("CG iterations: ", length(res_cg) - 1)
println("PCG identity iterations: ", length(res_pcg_id) - 1)
println("PCG MR-SPAI iterations: ", length(res_pcg_mr) - 1)

println("\nFinal relative residuals")
println("CG final residual: ", cg_rel_res)
println("PCG identity final residual: ", pcg_id_rel_res)
println("PCG MR-SPAI final residual: ", pcg_mr_rel_res)

println("\nRelative solution errors")
println("CG relative error: ", cg_rel_err)
println("PCG identity relative error: ", pcg_id_rel_err)
println("PCG MR-SPAI relative error: ", pcg_mr_rel_err)