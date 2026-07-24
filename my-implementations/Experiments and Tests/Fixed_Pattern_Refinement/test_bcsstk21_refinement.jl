using MatrixDepot
using SparseArrays
using LinearAlgebra
using Printf


# Paths

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

# Frobenius inner product, norm, and dropping
include(joinpath(utils_dir, "frobenius.jl"))
include(joinpath(utils_dir, "dropping.jl"))

# Float64 SPAI implementations
include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))

# Float32 SPAI implementations with optional dropping
include(joinpath(float32_dir, "my_pmr_float32_spai.jl"))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl"))
include(joinpath(float32_dir, "my_lopmr_float32_spai.jl"))

# Float64 fixed-pattern least-squares refinement
include(joinpath(spai_dir, "refine_spai_pattern.jl"))


# Helper functions

"""
    relative_spai_residual(A, M)

Computes

    ‖I - AM‖F / ‖I‖F

using Float64 arithmetic.
"""
function relative_spai_residual(
    A::SparseMatrixCSC,
    M::SparseMatrixCSC,
)
    n = size(A, 1)

    A64 = sparse(Float64.(A))
    M64 = sparse(Float64.(M))
    I64 = spdiagm(0 => ones(Float64, n))

    R = I64 - A64 * M64

    return frob_norm(R) / sqrt(Float64(n))
end


"""
    matrix_density(M)

Returns nnz(M) / length(M).
"""
function matrix_density(M::SparseMatrixCSC)
    return nnz(M) / length(M)
end


"""
Checks that the refined matrix contains no nonzero position outside
the original dropped Float32 sparsity pattern.
"""
function pattern_is_subset(
    M_refined::SparseMatrixCSC,
    M_pattern::SparseMatrixCSC,
)
    size(M_refined) == size(M_pattern) || return false

    pattern_rows = rowvals(M_pattern)
    refined_rows = rowvals(M_refined)

    for j in axes(M_pattern, 2)
        allowed_rows = Set(
            pattern_rows[nzrange(M_pattern, j)],
        )

        for position in nzrange(M_refined, j)
            refined_row = refined_rows[position]

            if refined_row ∉ allowed_rows
                return false
            end
        end
    end

    return true
end


# Fully Float64 experiment

function test_float64_baseline(
    method_name::String,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)
    println()
    println("$method_name: fully Float64")

    construction_time = @elapsed begin
        M64, residual_history = spai_method(
            A,
            Pr,
            M0,
            itmax,
            tol,
        )
    end

    final_residual = relative_spai_residual(A, M64)
    density = matrix_density(M64)

    println()
    @printf("Relative residual:         %.6e\n", final_residual)
    println("nnz(M):                    ", nnz(M64))
    @printf("Density of M:              %.6e\n", density)
    @printf("Construction time:         %.4f seconds\n", construction_time)

    return (
        method = method_name,
        M64 = M64,
        residual_history = residual_history,
        residual = final_residual,
        nnz_M = nnz(M64),
        density = density,
        construction_time = construction_time,
    )
end


# Float32 with dropping plus Float64 refinement

function test_refinement(
    method_name::String,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
    max_per_column::Int,
)
    n = size(A, 1)
    maximum_allowed_nnz = n * max_per_column

    println()
    println("$method_name: Float32 dropping plus Float64 refinement")

    println("Maximum entries per column: ", max_per_column)
    println("Maximum allowed nnz(M):      ", maximum_allowed_nnz)
    println()
    println("Computing dropped Float32 SPAI...")

    float32_time = @elapsed begin
        M32, residual_history = spai_method(
            A,
            Pr,
            M0,
            itmax,
            tol;
            max_per_column = max_per_column,
        )
    end

    residual_before = relative_spai_residual(A, M32)
    nnz_before = nnz(M32)
    density_before = matrix_density(M32)

    if nnz_before > maximum_allowed_nnz
        @warn(
            "The dropped Float32 matrix exceeds the requested nnz cap.",
            nnz_before = nnz_before,
            maximum_allowed_nnz = maximum_allowed_nnz,
        )
    end

    println("Refining dropped sparsity pattern in Float64...")

    refinement_time = @elapsed begin
        M_refined = refine_spai_pattern(A, M32)
    end

    residual_after = relative_spai_residual(A, M_refined)
    nnz_after = nnz(M_refined)
    density_after = matrix_density(M_refined)

    improvement_factor =
        residual_after == 0.0 ?
        Inf :
        residual_before / residual_after

    pattern_valid = pattern_is_subset(
        M_refined,
        M32,
    )

    println()
    @printf("Dropped Float32 residual:  %.6e\n", residual_before)
    @printf("Refined residual:          %.6e\n", residual_after)
    @printf("Improvement factor:        %.6e\n", improvement_factor)

    println()
    println("Maximum allowed nnz(M):    ", maximum_allowed_nnz)
    println("Float32 nnz(M):            ", nnz_before)
    println("Refined nnz(M):            ", nnz_after)
    @printf("Float32 density:           %.6e\n", density_before)
    @printf("Refined density:           %.6e\n", density_after)
    println("Pattern valid:             ", pattern_valid)

    println()
    @printf("Float32 construction time: %.4f seconds\n", float32_time)
    @printf("Float64 refinement time:   %.4f seconds\n", refinement_time)
    @printf(
        "Combined time:             %.4f seconds\n",
        float32_time + refinement_time,
    )

    return (
        method = method_name,
        M32 = M32,
        M_refined = M_refined,
        residual_history = residual_history,
        residual_before = residual_before,
        residual_after = residual_after,
        improvement_factor = improvement_factor,
        max_per_column = max_per_column,
        maximum_allowed_nnz = maximum_allowed_nnz,
        nnz_before = nnz_before,
        nnz_after = nnz_after,
        density_before = density_before,
        density_after = density_after,
        pattern_valid = pattern_valid,
        float32_time = float32_time,
        refinement_time = refinement_time,
        total_time = float32_time + refinement_time,
    )
end


# Loading bcsstk21

matrix_name = "bcsstk21"

md = mdopen("HB/bcsstk21")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("Matrix:          ", matrix_name)
println("Size:            ", size(A))
println("nnz(A):          ", nnz(A))
println("Symmetry error:  ", norm(A - transpose(A)))


# Experiment parameters

itmax = 300
tol = 1.0e-15
max_per_column = 200

maximum_allowed_nnz = n * max_per_column
maximum_density = maximum_allowed_nnz / n^2

println()
println("Dropping parameters")
println("Maximum entries per column: ", max_per_column)
println("Maximum total nnz(M):        ", maximum_allowed_nnz)
@printf("Maximum possible density:    %.6e\n", maximum_density)

A_diagonal = diag(A)

if any(iszero, A_diagonal)
    error(
        "Jacobi preconditioner cannot be formed because " *
        "A has a zero diagonal entry.",
    )
end


# Float64 initial inverse and Jacobi preconditioner

M0_64 = spdiagm(
    0 => ones(Float64, n),
)

Pr64 = spdiagm(
    0 => 1.0 ./ A_diagonal,
)


# Float32 initial inverse and Jacobi preconditioner

M0_32 = spdiagm(
    0 => ones(Float32, n),
)

Pr32 = spdiagm(
    0 => Float32.(1.0 ./ A_diagonal),
)


# Fully Float64 baseline runs

pmr_float64 = test_float64_baseline(
    "PMR",
    my_pmr,
    A,
    Pr64,
    M0_64,
    itmax,
    tol,
)

pcg_float64 = test_float64_baseline(
    "PCG",
    my_pcg,
    A,
    Pr64,
    M0_64,
    itmax,
    tol,
)

lopmr_float64 = test_float64_baseline(
    "LOPMR",
    my_lopmr,
    A,
    Pr64,
    M0_64,
    itmax,
    tol,
)


# Float32 dropping plus Float64 refinement runs

pmr_refined = test_refinement(
    "PMR",
    my_pmr_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
    max_per_column,
)

pcg_refined = test_refinement(
    "PCG",
    my_pcg_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
    max_per_column,
)

lopmr_refined = test_refinement(
    "LOPMR",
    my_lopmr_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
    max_per_column,
)


# Final residual summary

println()
println("="^120)
println(
    "FINAL RESIDUAL SUMMARY: bcsstk21, " *
    "max_per_column = $max_per_column",
)
println("="^120)

@printf(
    "%-10s %-16s %-16s %-16s %-14s\n",
    "Method",
    "Float64",
    "Float32 drop",
    "Refined",
    "Improvement",
)

for (baseline, refined) in (
    (pmr_float64, pmr_refined),
    (pcg_float64, pcg_refined),
    (lopmr_float64, lopmr_refined),
)
    @printf(
        "%-10s %-16.6e %-16.6e %-16.6e %-14.4e\n",
        baseline.method,
        baseline.residual,
        refined.residual_before,
        refined.residual_after,
        refined.improvement_factor,
    )
end


# Final time and sparsity summary

println()
println("="^135)
println(
    "TIME AND SPARSITY SUMMARY: bcsstk21, " *
    "max_per_column = $max_per_column",
)
println("="^135)

@printf(
    "%-10s %-13s %-13s %-13s %-13s %-13s %-13s %-13s\n",
    "Method",
    "F64 time",
    "F32 time",
    "Refine",
    "Mixed total",
    "F64 nnz",
    "F32 nnz",
    "Refined nnz",
)

for (baseline, refined) in (
    (pmr_float64, pmr_refined),
    (pcg_float64, pcg_refined),
    (lopmr_float64, lopmr_refined),
)
    @printf(
        "%-10s %-13.4f %-13.4f %-13.4f %-13.4f %-13d %-13d %-13d\n",
        baseline.method,
        baseline.construction_time,
        refined.float32_time,
        refined.refinement_time,
        refined.total_time,
        baseline.nnz_M,
        refined.nnz_before,
        refined.nnz_after,
    )
end


# Final density summary

println()
println("DENSITY SUMMARY")

@printf(
    "%-10s %-18s %-18s %-18s\n",
    "Method",
    "Float64 density",
    "Float32 density",
    "Refined density",
)

for (baseline, refined) in (
    (pmr_float64, pmr_refined),
    (pcg_float64, pcg_refined),
    (lopmr_float64, lopmr_refined),
)
    @printf(
        "%-10s %-18.6e %-18.6e %-18.6e\n",
        baseline.method,
        baseline.density,
        refined.density_before,
        refined.density_after,
    )
end