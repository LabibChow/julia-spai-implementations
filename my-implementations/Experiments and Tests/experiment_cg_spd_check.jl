using MatrixDepot
using SparseArrays
using LinearAlgebra

# This experiment checks whether the selected matrix is symmetric positive definite (SPD).
include(joinpath(@__DIR__, "..", "Linear Solvers", "my_cg_solve.jl"))
include(joinpath(@__DIR__, "..", "Utils", "check_spd.jl"))

md = mdopen("HB/nos1")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("======")
println("Matrix: HB/nos1")
println("Size: ", size(A))
println("Nonzeros in A: ", nnz(A))
println("Element type: ", eltype(A))
println("=====")


# 2. Check whether the original matrix A is SPD
# check_spd calculates:
# - relative symmetry error
# - smallest eigenvalue
# - largest eigenvalue
# - whether Cholesky succeeds

println("\nChecking whether HB/nos1 is SPD...")

spd_result_A = check_spd(A; name = "HB/nos1")

# 3. Stop if A is not SPD

if !spd_result_A.is_spd
    println("\nHB/nos1 did not pass the SPD checks.")
    println("CG experiment will not be performed.")

    error("Experiment stopped because HB/nos1 is not SPD.")
end

println("\nHB/nos1 passed the SPD checks.")
println("Proceeding with the CG experiment...")


# 4. Create the test problem Ax = b

x_exact = ones(Float64, n)
b = A * x_exact
x0 = zeros(Float64, n)

itmax = 1000
tol = 1e-5

println("\nTest problem created.")
println("Maximum CG iterations: ", itmax)
println("CG tolerance: ", tol)


# 5. Run normal CG

println("\nRunning CG without a preconditioner...")

x_cg, res_cg = my_cg_solve(
    A,
    b,
    x0,
    itmax,
    tol
)

# 6. Calculate CG results

cg_iterations = length(res_cg) - 1

cg_rel_res =
    norm(b - A * x_cg) / norm(b)

cg_rel_err =
    norm(x_cg - x_exact) / norm(x_exact)

cg_converged =
    cg_rel_res < tol

println("\nMatrix information")
println("Matrix: HB/nos1")
println("Size: ", size(A))
println("Nonzeros: ", nnz(A))

println("\nSPD checks")
println("Symmetry error: ", spd_result_A.symmetry_error)
println("Smallest eigenvalue: ", spd_result_A.lambda_min)
println("Largest eigenvalue: ", spd_result_A.lambda_max)
println("Cholesky succeeded: ", spd_result_A.cholesky_success)
println("Classified as SPD: ", spd_result_A.is_spd)

println("\nCG results")
println("Iterations: ", cg_iterations)
println("Initial recorded residual: ", first(res_cg))
println("Final recorded residual: ", last(res_cg))
println("Final directly calculated residual: ", cg_rel_res)
println("Relative solution error: ", cg_rel_err)
println("Converged: ", cg_converged)