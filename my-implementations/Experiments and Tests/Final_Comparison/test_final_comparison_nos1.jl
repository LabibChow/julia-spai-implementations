using MatrixDepot
using SparseArrays
using LinearAlgebra
using Plots

include("../../Utils/check_spd.jl")
include("../../Linear Solvers/my_cg_solve.jl")
include("../../Linear Solvers/my_pcg_solve.jl")

include("../../SPAI Iterations/my_cg_spai.jl")
include("../../SPAI Iterations/my_mr_spai.jl")
include("../../SPAI Iterations/my_lomr_spai.jl")

# Final comparison on HB/nos1
# CG-SPAI vs MR-SPAI vs LOMR-SPAI

md = mdopen("HB/nos1")

A = sparse(Float64.(md.A))
n = size(A, 1)

println("=====")
println("FINAL COMPARISON")
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

# 2. Common settings

spai_itmax = 40
spai_tol = 1e-3

solver_itmax = 1000
solver_tol = 1e-5

I_sparse = spdiagm(0 => ones(Float64, n))

# 3. Create Ax = b

x_exact = ones(Float64, n)
b = A * x_exact
x0 = zeros(Float64, n)

# 4. Baseline CG

println("\nRunning baseline CG...")

x_cg, res_cg =
    my_cg_solve(A, b, x0, solver_itmax, solver_tol)

cg_rel_res =
    norm(b - A * x_cg) / norm(b)

cg_rel_err =
    norm(x_cg - x_exact) / norm(x_exact)

# 5. PCG identity

println("\nRunning PCG with identity preconditioner...")

M_identity = spdiagm(0 => ones(Float64, n))

x_pcg_id, res_pcg_id =
    my_pcg_solve(A, b, x0, M_identity, solver_itmax, solver_tol)

pcg_id_rel_res =
    norm(b - A * x_pcg_id) / norm(b)

pcg_id_rel_err =
    norm(x_pcg_id - x_exact) / norm(x_exact)

# 6. Build CG-SPAI and check SPD

println("\nBuilding CG-SPAI...")

M0_cg = spzeros(Float64, n, n)

M_cg_spai, res_cg_spai =
    my_cg(A, M0_cg, spai_itmax, spai_tol)

cg_spai_quality =
    norm(I_sparse - A * M_cg_spai) / norm(I_sparse)

spd_cg_spai =
    check_spd(M_cg_spai; name = "CG-SPAI")

# 7. Build MR-SPAI and check SPD

println("\nBuilding MR-SPAI...")

M0_mr = spzeros(Float64, n, n)

M_mr, res_mr =
    my_mr(A, M0_mr, spai_itmax, spai_tol)

mr_spai_quality =
    norm(I_sparse - A * M_mr) / norm(I_sparse)

spd_mr =
    check_spd(M_mr; name = "MR-SPAI")

# 8. Build LOMR-SPAI and check SPD

println("\nBuilding LOMR-SPAI...")

M0_lomr = spzeros(Float64, n, n)

M_lomr, res_lomr =
    my_lomr(A, M0_lomr, spai_itmax, spai_tol)

lomr_spai_quality =
    norm(I_sparse - A * M_lomr) / norm(I_sparse)

spd_lomr =
    check_spd(M_lomr; name = "LOMR-SPAI")

# 9. Run PCG with MR-SPAI if SPD

if spd_mr.is_spd
    println("\nRunning PCG with MR-SPAI...")

    x_pcg_mr, res_pcg_mr =
        my_pcg_solve(A, b, x0, M_mr, solver_itmax, solver_tol)

    pcg_mr_rel_res =
        norm(b - A * x_pcg_mr) / norm(b)

    pcg_mr_rel_err =
        norm(x_pcg_mr - x_exact) / norm(x_exact)
else
    res_pcg_mr = Float64[]
    pcg_mr_rel_res = NaN
    pcg_mr_rel_err = NaN
end

# 10. Run PCG with LOMR-SPAI if SPD

if spd_lomr.is_spd
    println("\nRunning PCG with LOMR-SPAI...")

    x_pcg_lomr, res_pcg_lomr =
        my_pcg_solve(A, b, x0, M_lomr, solver_itmax, solver_tol)

    pcg_lomr_rel_res =
        norm(b - A * x_pcg_lomr) / norm(b)

    pcg_lomr_rel_err =
        norm(x_pcg_lomr - x_exact) / norm(x_exact)
else
    res_pcg_lomr = Float64[]
    pcg_lomr_rel_res = NaN
    pcg_lomr_rel_err = NaN
end

# 11. Plot SPAI construction convergence

p1 = plot(
    0:length(res_cg_spai)-1,
    res_cg_spai,
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "SPAI residual",
    title = "SPAI construction comparison for HB/nos1",
    label = "CG-SPAI",
    linewidth = 3
)

plot!(
    p1,
    0:length(res_mr)-1,
    res_mr,
    label = "MR-SPAI",
    linestyle = :dash,
    linewidth = 3
)

plot!(
    p1,
    0:length(res_lomr)-1,
    res_lomr,
    label = "LOMR-SPAI",
    linestyle = :dot,
    linewidth = 3
)

savefig(p1, "nos1_spai_method_comparison.png")

# 12. Plot solver convergence

p2 = plot(
    0:length(res_cg)-1,
    res_cg,
    yscale = :log10,
    xlabel = "Solver iteration",
    ylabel = "Relative residual",
    title = "Solver convergence comparison for HB/nos1",
    label = "CG",
    linewidth = 3
)

plot!(
    p2,
    0:length(res_pcg_id)-1,
    res_pcg_id,
    label = "PCG Identity",
    linestyle = :dash,
    linewidth = 2
)

if spd_mr.is_spd
    plot!(
        p2,
        0:length(res_pcg_mr)-1,
        res_pcg_mr,
        label = "PCG MR-SPAI",
        linestyle = :dot,
        linewidth = 3
    )
end

if spd_lomr.is_spd
    plot!(
        p2,
        0:length(res_pcg_lomr)-1,
        res_pcg_lomr,
        label = "PCG LOMR-SPAI",
        linestyle = :dashdot,
        linewidth = 3
    )
end

savefig(p2, "nos1_solver_method_comparison.png")

# 13. Summary

println("\n=====")
println("SUMMARY")
println("=====")

println("\nSPAI construction results")
println("CG-SPAI quality: ", cg_spai_quality)
println("CG-SPAI smallest eigenvalue: ", spd_cg_spai.lambda_min)
println("CG-SPAI SPD: ", spd_cg_spai.is_spd)

println("\nMR-SPAI quality: ", mr_spai_quality)
println("MR-SPAI smallest eigenvalue: ", spd_mr.lambda_min)
println("MR-SPAI SPD: ", spd_mr.is_spd)

println("\nLOMR-SPAI quality: ", lomr_spai_quality)
println("LOMR-SPAI smallest eigenvalue: ", spd_lomr.lambda_min)
println("LOMR-SPAI SPD: ", spd_lomr.is_spd)

println("\nSolver iteration counts")
println("CG iterations: ", length(res_cg) - 1)
println("PCG identity iterations: ", length(res_pcg_id) - 1)

if spd_mr.is_spd
    println("PCG MR-SPAI iterations: ", length(res_pcg_mr) - 1)
else
    println("PCG MR-SPAI iterations: skipped")
end

if spd_lomr.is_spd
    println("PCG LOMR-SPAI iterations: ", length(res_pcg_lomr) - 1)
else
    println("PCG LOMR-SPAI iterations: skipped")
end

println("\nFinal relative residuals")
println("CG final residual: ", cg_rel_res)
println("PCG identity final residual: ", pcg_id_rel_res)

if spd_mr.is_spd
    println("PCG MR-SPAI final residual: ", pcg_mr_rel_res)
end

if spd_lomr.is_spd
    println("PCG LOMR-SPAI final residual: ", pcg_lomr_rel_res)
end

println("\nRelative solution errors")
println("CG relative error: ", cg_rel_err)
println("PCG identity relative error: ", pcg_id_rel_err)

if spd_mr.is_spd
    println("PCG MR-SPAI relative error: ", pcg_mr_rel_err)
end

if spd_lomr.is_spd
    println("PCG LOMR-SPAI relative error: ", pcg_lomr_rel_err)
end

println("\nSaved plots:")
println("nos1_spai_method_comparison.png")
println("nos1_solver_method_comparison.png")