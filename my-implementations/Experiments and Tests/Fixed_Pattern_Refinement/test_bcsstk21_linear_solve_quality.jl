using MatrixDepot
using SparseArrays
using LinearAlgebra
using Random
using Printf


spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)

solver_dir = joinpath(@__DIR__, "..", "..", "Linear Solvers",)

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)


# Includes

include(joinpath(utils_dir, "frobenius.jl"))
include(joinpath(utils_dir, "dropping.jl"))

include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))

include(joinpath(solver_dir, "my_pcg_solve.jl"))


# Helper function

function test_linear_solve(
    preconditioner_name::String,
    A,
    b,
    x0,
    M,
    itmax,
    tol,
)
    println()
    println("Preconditioner: ", preconditioner_name)

    solve_time = @elapsed begin
        x, residual_history = my_pcg_solve(
            A,
            b,
            x0,
            M,
            itmax,
            tol,
        )
    end

    iterations = length(residual_history) - 1
    final_residual = residual_history[end]

    true_relative_residual =
        norm(b - A * x) / norm(b)

    println("Iterations:               ", iterations)
    @printf("Recorded residual:        %.6e\n", final_residual)
    @printf("True relative residual:   %.6e\n", true_relative_residual)
    @printf("Solve time:               %.6f seconds\n", solve_time)
    println("Converged:                ", final_residual < tol)

    return (
        name = preconditioner_name,
        x = x,
        residual_history = residual_history,
        iterations = iterations,
        final_residual = final_residual,
        true_relative_residual = true_relative_residual,
        solve_time = solve_time,
        converged = final_residual < tol,
    )
end


# Loading the matrix bcsstk21

md = mdopen("HB/bcsstk21")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("Matrix: bcsstk21")
println("Size:   ", size(A))
println("nnz(A): ", nnz(A))


# SPAI construction parameters

spai_itmax = 300
spai_tol = 1.0e-15
max_per_column = 200

A_diagonal = diag(A)

Pr64 = spdiagm(
    0 => 1.0 ./ A_diagonal,
)

Pr32 = spdiagm(
    0 => Float32.(1.0 ./ A_diagonal),
)

M0_64 = spdiagm(
    0 => ones(Float64, n),
)

M0_32 = spdiagm(
    0 => ones(Float32, n),
)


# Construct PCG SPAIs

println()
println("Constructing fully Float64 PCG SPAI...")

float64_spai_time = @elapsed begin
    M_float64, _ = my_pcg(
        A,
        Pr64,
        M0_64,
        spai_itmax,
        spai_tol,
    )
end

println("Constructing dropped Float32 PCG SPAI...")

float32_spai_time = @elapsed begin
    M_float32, _ = my_pcg_float32(
        A,
        Pr32,
        M0_32,
        spai_itmax,
        spai_tol;
        max_per_column = max_per_column,
    )
end

println("Refining the dropped Float32 pattern in Float64...")

refinement_time = @elapsed begin
    M_refined = refine_spai_pattern(
        A,
        M_float32,
    )
end


# Converting the Float32 preconditioner because my_pcg_solve requires
# A, b, x0, and M to have the same floating-point type.
M_float32_64 = sparse(Float64.(M_float32))

# Identity represents the unpreconditioned comparison.
M_identity = spdiagm(
    0 => ones(Float64, n),
)


# Linear system

Random.seed!(1)

b = randn(Float64, n)
x0 = zeros(Float64, n)

solve_itmax = 5000
solve_tol = 1.0e-8


# Solve comparisons

identity_result = test_linear_solve(
    "Identity / no preconditioner",
    A,
    b,
    x0,
    M_identity,
    solve_itmax,
    solve_tol,
)

float64_result = test_linear_solve(
    "Float64 PCG SPAI",
    A,
    b,
    x0,
    M_float64,
    solve_itmax,
    solve_tol,
)

float32_result = test_linear_solve(
    "Dropped Float32 PCG SPAI",
    A,
    b,
    x0,
    M_float32_64,
    solve_itmax,
    solve_tol,
)

refined_result = test_linear_solve(
    "Refined Float32 pattern",
    A,
    b,
    x0,
    M_refined,
    solve_itmax,
    solve_tol,
)


# Final solve-quality summary

println()
println("LINEAR SOLVE QUALITY SUMMARY")

@printf(
    "%-32s %-12s %-18s %-15s %-12s\n",
    "Preconditioner",
    "Iterations",
    "Final residual",
    "Solve time",
    "Converged",
)

for result in (
    identity_result,
    float64_result,
    float32_result,
    refined_result,
)
    @printf(
        "%-32s %-12d %-18.6e %-15.6f %-12s\n",
        result.name,
        result.iterations,
        result.final_residual,
        result.solve_time,
        string(result.converged),
    )
end


# SPAI construction summary

println()
println("SPAI CONSTRUCTION SUMMARY")

@printf(
    "%-28s %-16s %-16s\n",
    "SPAI",
    "Construction time",
    "nnz(M)",
)

@printf(
    "%-28s %-16.4f %-16d\n",
    "Float64 PCG SPAI",
    float64_spai_time,
    nnz(M_float64),
)

@printf(
    "%-28s %-16.4f %-16d\n",
    "Dropped Float32 SPAI",
    float32_spai_time,
    nnz(M_float32),
)

@printf(
    "%-28s %-16.4f %-16d\n",
    "Float64 refinement",
    refinement_time,
    nnz(M_refined),
)