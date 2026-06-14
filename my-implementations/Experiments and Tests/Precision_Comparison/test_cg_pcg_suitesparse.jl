using MatrixDepot
using SparseArrays
using LinearAlgebra
using Plots

include("../../Utils/check_spd.jl")
include("../../Linear Solvers/my_cg_solve.jl")
include("../../Linear Solvers/my_pcg_solve.jl")
include("../../SPAI Iterations/my_cg_spai.jl")

# 1. Load SuiteSparse matrix

md = mdopen("HB/nos1")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("RUNNING VERSION WITH SPAI SPD CHECK")

println("======================================")
println("Matrix: HB/nos1")
println("Size: ", size(A))
println("Nonzeros in A: ", nnz(A))
println("======================================")

# 2. Create Ax = b problem
# We choose x_exact = ones(n), so b = A*x_exact.
# This way, we know the true solution.

x_exact = ones(Float64, n)
b = A * x_exact
x0 = zeros(Float64, n)

itmax = 1000
tol = 1e-5

# 3. Running normal CG: Ax = b

println("\nRunning CG without preconditioner...")

x_cg, res_cg = my_cg_solve(A, b, x0, itmax, tol)

cg_rel_res = norm(b - A * x_cg) / norm(b)
cg_rel_err = norm(x_cg - x_exact) / norm(x_exact)

println("\nCG Results")
println("Final relative residual: ", cg_rel_res)
println("Relative error: ", cg_rel_err)
println("Iterations: ", length(res_cg) - 1)

# 4. Run PCG with identity preconditioner for sanity check, should be similar to normal CG.

M_identity = spdiagm(0 => ones(Float64, n))

println("\nRunning PCG with identity preconditioner...")

x_pcg_id, res_pcg_id = my_pcg_solve(A, b, x0, M_identity, itmax, tol)

pcg_id_rel_res = norm(b - A * x_pcg_id) / norm(b)
pcg_id_rel_err = norm(x_pcg_id - x_exact) / norm(x_exact)

println("\nPCG Identity Results")
println("Final relative residual: ", pcg_id_rel_res)
println("Relative error: ", pcg_id_rel_err)
println("Iterations: ", length(res_pcg_id) - 1)

# 5. Build SPAI preconditioner using my_cg_spai.jl
# This tries to find M such that A*M ≈ I.

println("\nBuilding SPAI preconditioner using my_cg_spai...")

spai_itmax = 40
spai_tol = 1e-3

M0 = spzeros(Float64, n, n)

M_spai, spai_res = my_cg(A, M0, spai_itmax, spai_tol)

# Plot SPAI residual convergence

plot(
    0:length(spai_res)-1,
    spai_res,
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "SPAI residual",
    title = "CG-SPAI convergence for HB/nos1",
    label = "Float64"
)

savefig("nos1_spai_convergence.png")

println("\nSPAI construction done.")
println("SPAI iterations: ", length(spai_res) - 1)
println("Nonzeros in M_spai: ", nnz(M_spai))

# Calculate SPAI approximation quality
I_sparse = spdiagm(0 => ones(Float64, n))
R_spai = I_sparse - A * M_spai

spai_quality =
    norm(R_spai) / norm(I_sparse)

println(
    "SPAI quality ||I - A*M||_F / ||I||_F: ",
    spai_quality
)

# Check whether the CG-based SPAI is SPD

println("\nChecking whether M_spai is SPD...")

spd_result_M = check_spd(M_spai; name = "CG-based SPAI")

# Plot spectrum of M_spai

eigenvalues_M = sort(spd_result_M.eigenvalues)

plot(
    1:length(eigenvalues_M),
    eigenvalues_M,
    xlabel = "Eigenvalue index",
    ylabel = "Eigenvalue",
    title = "Spectrum of CG-SPAI for HB/nos1",
    label = "Eigenvalues"
)

hline!(
    [0.0],
    linestyle = :dash,
    label = "Zero"
)

savefig("nos1_spai_spectrum.png")

println("\nSPAI SPD summary")
println("Symmetry error: ", spd_result_M.symmetry_error)
println("Smallest eigenvalue: ", spd_result_M.lambda_min)
println("Largest eigenvalue: ", spd_result_M.lambda_max)
println("Cholesky succeeded: ", spd_result_M.cholesky_success)
println("Classified as SPD: ", spd_result_M.is_spd)


# 6. Run PCG only if the SPAI is SPD

if spd_result_M.is_spd

    println("\nM_spai passed the SPD checks.")
    println("Running PCG with SPAI preconditioner...")

    x_pcg_spai, res_pcg_spai =
        my_pcg_solve(A, b, x0, M_spai, itmax, tol)

    pcg_spai_rel_res =
        norm(b - A * x_pcg_spai) / norm(b)

    pcg_spai_rel_err =
        norm(x_pcg_spai - x_exact) / norm(x_exact)

    println("\nPCG SPAI Results")
    println("Final relative residual: ", pcg_spai_rel_res)
    println("Relative error: ", pcg_spai_rel_err)
    println("Iterations: ", length(res_pcg_spai) - 1)

else
    println("\nM_spai did not pass the SPD checks.")
    println("PCG with this SPAI will not be run.")

    x_pcg_spai = nothing
    res_pcg_spai = Float64[]
    pcg_spai_rel_res = NaN
    pcg_spai_rel_err = NaN
end

# 7. Summary
println("\n=====")
println("SUMMARY")
println("=====")

println("\nSolver iteration counts")
println("CG iterations: ", length(res_cg) - 1)
println("PCG identity iterations: ", length(res_pcg_id) - 1)

if spd_result_M.is_spd
    println("PCG SPAI iterations: ", length(res_pcg_spai) - 1)
else
    println("PCG SPAI iterations: skipped")
end

println("\nFinal relative residuals")
println("CG final residual: ", cg_rel_res)
println("PCG identity final residual: ", pcg_id_rel_res)

if spd_result_M.is_spd
    println("PCG SPAI final residual: ", pcg_spai_rel_res)
else
    println("PCG SPAI final residual: not available")
end

println("\nRelative solution errors")
println("CG relative error: ", cg_rel_err)
println("PCG identity relative error: ", pcg_id_rel_err)

if spd_result_M.is_spd
    println("PCG SPAI relative error: ", pcg_spai_rel_err)
else
    println("PCG SPAI relative error: not available")
end

println("\nSPAI information")
println("SPAI quality ||I - A*M||_F / ||I||_F: ", spai_quality)
println("SPAI symmetry error: ", spd_result_M.symmetry_error)
println("SPAI smallest eigenvalue: ", spd_result_M.lambda_min)
println("SPAI largest eigenvalue: ", spd_result_M.lambda_max)
println("SPAI Cholesky succeeded: ", spd_result_M.cholesky_success)
println("SPAI classified as SPD: ", spd_result_M.is_spd)

if !spd_result_M.is_spd
    println("PCG with SPAI was skipped because the SPAI was not SPD.")
end