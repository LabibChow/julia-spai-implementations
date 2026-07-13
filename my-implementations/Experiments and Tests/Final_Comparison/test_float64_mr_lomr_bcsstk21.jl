using MatrixDepot
using SparseArrays
using LinearAlgebra
using Plots

include(joinpath(@__DIR__, "..", "..", "Utils", "check_spd.jl"))
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_cg_solve.jl"))
include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_pcg_solve.jl"))

include(joinpath(@__DIR__, "..", "..", "SPAI Iterations", "my_mr_spai.jl"))
include(joinpath(@__DIR__, "..", "..", "SPAI Iterations", "my_lomr_spai.jl"))

# Float64 MR/LOMR-SPAI + Float64 PCG
# Matrix: HB/bcsstk21

md = mdopen("HB/bcsstk21")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("FLOAT64 MR/LOMR-SPAI + FLOAT64 PCG")
println("Matrix: HB/bcsstk21")
println("Size: ", size(A))
println("Nonzeros in A: ", nnz(A))

# 1. Check original matrix

println("\nChecking whether original matrix A is SPD...")

spd_A = check_spd(A; name = "HB/bcsstk21")

if !spd_A.is_spd
    error("HB/bcsstk21 is not SPD, stopping experiment.")
end

# 2. Settings

spai_itmax = 40
spai_tol = 1e-3

solver_itmax = 1000
solver_tol = 1e-5

I_sparse = spdiagm(0 => ones(Float64, n))

# 3. Build MR-SPAI Float64

println("\nBuilding Float64 MR-SPAI...")

M0_mr = spzeros(Float64, n, n)

M_mr, res_mr =
    my_mr(A, M0_mr, spai_itmax, spai_tol)

mr_quality =
    norm(I_sparse - A * M_mr) / norm(I_sparse)

println("\nChecking MR-SPAI Float64...")
spd_mr =
    check_spd(M_mr; name = "MR-SPAI Float64")

# 4. Build LOMR-SPAI Float64

println("\nBuilding Float64 LOMR-SPAI...")

M0_lomr = spzeros(Float64, n, n)

M_lomr, res_lomr =
    my_lomr(A, M0_lomr, spai_itmax, spai_tol)

lomr_quality =
    norm(I_sparse - A * M_lomr) / norm(I_sparse)

println("\nChecking LOMR-SPAI Float64...")
spd_lomr =
    check_spd(M_lomr; name = "LOMR-SPAI Float64")

# 5. Create Ax = b

x_exact = ones(Float64, n)
b = A * x_exact
x0 = zeros(Float64, n)

# 6. Baseline CG

println("\nRunning baseline Float64 CG...")

x_cg, res_cg =
    my_cg_solve(A, b, x0, solver_itmax, solver_tol)

cg_rel_res =
    norm(b - A * x_cg) / norm(b)

cg_rel_err =
    norm(x_cg - x_exact) / norm(x_exact)

# 7. PCG with MR-SPAI if SPD

if spd_mr.is_spd

    println("\nRunning PCG with Float64 MR-SPAI...")

    x_pcg_mr, res_pcg_mr =
        my_pcg_solve(A, b, x0, M_mr, solver_itmax, solver_tol)

    pcg_mr_rel_res =
        norm(b - A * x_pcg_mr) / norm(b)

    pcg_mr_rel_err =
        norm(x_pcg_mr - x_exact) / norm(x_exact)

else
    println("\nMR-SPAI Float64 is not SPD. PCG skipped.")

    res_pcg_mr = Float64[]
    pcg_mr_rel_res = NaN
    pcg_mr_rel_err = NaN
end

# 8. PCG with LOMR-SPAI if SPD

if spd_lomr.is_spd

    println("\nRunning PCG with Float64 LOMR-SPAI...")

    x_pcg_lomr, res_pcg_lomr =
        my_pcg_solve(A, b, x0, M_lomr, solver_itmax, solver_tol)

    pcg_lomr_rel_res =
        norm(b - A * x_pcg_lomr) / norm(b)

    pcg_lomr_rel_err =
        norm(x_pcg_lomr - x_exact) / norm(x_exact)

else
    println("\nLOMR-SPAI Float64 is not SPD. PCG skipped.")

    res_pcg_lomr = Float64[]
    pcg_lomr_rel_res = NaN
    pcg_lomr_rel_err = NaN
end

# 9. Plot SPAI construction convergence

p1 = plot(
    0:length(res_mr)-1,
    res_mr,
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "SPAI residual",
    title = "Float64 MR/LOMR-SPAI construction for bcsstk21",
    label = "MR-SPAI Float64",
    linewidth = 3
)

plot!(
    p1,
    0:length(res_lomr)-1,
    res_lomr,
    label = "LOMR-SPAI Float64",
    linestyle = :dash,
    linewidth = 3
)

savefig(p1, "bcsstk21_float64_mr_lomr_spai_convergence.png")

# 10. Plot solver convergence

p2 = plot(
    0:length(res_cg)-1,
    res_cg,
    yscale = :log10,
    xlabel = "Solver iteration",
    ylabel = "Relative residual",
    title = "Float64 MR/LOMR-SPAI with PCG for bcsstk21",
    label = "CG baseline",
    linewidth = 3
)

if spd_mr.is_spd
    plot!(
        p2,
        0:length(res_pcg_mr)-1,
        res_pcg_mr,
        label = "PCG + MR-SPAI Float64",
        linestyle = :dash,
        linewidth = 3
    )
end

if spd_lomr.is_spd
    plot!(
        p2,
        0:length(res_pcg_lomr)-1,
        res_pcg_lomr,
        label = "PCG + LOMR-SPAI Float64",
        linestyle = :dot,
        linewidth = 3
    )
end

savefig(p2, "bcsstk21_float64_mr_lomr_pcg.png")

# 11. Summary

println("FLOAT64 BCSSTK21 SUMMARY")

println("\nBaseline Float64 CG")
println("Iterations: ", length(res_cg) - 1)
println("Final residual: ", cg_rel_res)
println("Relative error: ", cg_rel_err)

println("\nFloat64 MR-SPAI")
println("SPAI quality: ", mr_quality)
println("Symmetry error: ", spd_mr.symmetry_error)
println("Smallest eigenvalue: ", spd_mr.lambda_min)
println("Cholesky succeeded: ", spd_mr.cholesky_success)
println("SPD: ", spd_mr.is_spd)
println("Nonzeros in M: ", nnz(M_mr))

if spd_mr.is_spd
    println("PCG iterations: ", length(res_pcg_mr) - 1)
    println("PCG final residual: ", pcg_mr_rel_res)
    println("PCG relative error: ", pcg_mr_rel_err)
else
    println("PCG skipped because MR-SPAI Float64 was not SPD.")
end

println("\nFloat64 LOMR-SPAI")
println("SPAI quality: ", lomr_quality)
println("Symmetry error: ", spd_lomr.symmetry_error)
println("Smallest eigenvalue: ", spd_lomr.lambda_min)
println("Cholesky succeeded: ", spd_lomr.cholesky_success)
println("SPD: ", spd_lomr.is_spd)
println("Nonzeros in M: ", nnz(M_lomr))

if spd_lomr.is_spd
    println("PCG iterations: ", length(res_pcg_lomr) - 1)
    println("PCG final residual: ", pcg_lomr_rel_res)
    println("PCG relative error: ", pcg_lomr_rel_err)
else
    println("PCG skipped because LOMR-SPAI Float64 was not SPD.")
end

println("\nSaved plots:")
println("bcsstk21_float64_mr_lomr_spai_convergence.png")
println("bcsstk21_float64_mr_lomr_pcg.png")