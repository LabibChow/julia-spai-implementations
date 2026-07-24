using MatrixDepot
using SparseArrays
using LinearAlgebra
using Printf


# Paths

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)


include(joinpath(utils_dir, "frobenius.jl"))
include(joinpath(utils_dir, "dropping.jl"))

include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))


# Helper functions

"""
Computes ‖I - AM‖F / ‖I‖F in Float64.
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
Computes the relative symmetry error ‖M-Mᵀ‖F / ‖M‖F.
"""
function relative_symmetry_error(M::SparseMatrixCSC)
    M64 = sparse(Float64.(M))
    denominator = frob_norm(M64)

    if denominator == 0.0
        return 0.0
    end

    return frob_norm(M64 - transpose(M64)) / denominator
end


"""
Checks that refinement introduces no positions outside the original
dropped Float32 pattern.
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
            if refined_rows[position] ∉ allowed_rows
                return false
            end
        end
    end

    return true
end


# Loading msc04515

matrix_name = "msc04515"

md = mdopen("Boeing/msc04515")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("="^80)
println("Matrix:          ", matrix_name)
println("Size:            ", size(A))
println("nnz(A):          ", nnz(A))
println("Density of A:    ", nnz(A) / n^2)
println("Symmetry error:  ", norm(A - transpose(A)))
println("="^80)


# Experiment parameters

itmax = 300
tol = 1.0e-15
max_per_column = 100

maximum_allowed_nnz = n * max_per_column
maximum_density = maximum_allowed_nnz / n^2

println()
println("Experiment parameters")
println("itmax:                     ", itmax)
println("tol:                       ", tol)
println("Maximum entries/column:    ", max_per_column)
println("Maximum allowed nnz(M):    ", maximum_allowed_nnz)
@printf("Maximum allowed density:   %.6e\n", maximum_density)


# Initial matrices and Jacobi preconditioners

A_diagonal = diag(A)

if any(iszero, A_diagonal)
    error(
        "Jacobi preconditioner cannot be constructed because " *
        "A contains a zero diagonal entry.",
    )
end

M0_64 = spdiagm(
    0 => ones(Float64, n),
)

Pr64 = spdiagm(
    0 => 1.0 ./ A_diagonal,
)

M0_32 = spdiagm(
    0 => ones(Float32, n),
)

Pr32 = spdiagm(
    0 => Float32.(1.0 ./ A_diagonal),
)


# Fully Float64 PCG SPAI

println()
println("Fully Float64 PCG SPAI")

float64_time = @elapsed begin
    M_float64, residual_history64 = my_pcg(
        A,
        Pr64,
        M0_64,
        itmax,
        tol,
    )
end

float64_residual = relative_spai_residual(
    A,
    M_float64,
)

float64_symmetry = relative_symmetry_error(
    M_float64,
)

println()
@printf("Relative SPAI residual:    %.6e\n", float64_residual)
@printf("Relative symmetry error:   %.6e\n", float64_symmetry)
println("nnz(M):                    ", nnz(M_float64))
@printf("Density of M:              %.6e\n", nnz(M_float64) / n^2)
@printf("Construction time:         %.4f seconds\n", float64_time)


# Dropped Float32 PCG SPAI

println()
println("Dropped Float32 PCG SPAI")

float32_time = @elapsed begin
    M_float32, residual_history32 = my_pcg_float32(
        A,
        Pr32,
        M0_32,
        itmax,
        tol;
        max_per_column = max_per_column,
    )
end

float32_residual = relative_spai_residual(
    A,
    M_float32,
)

float32_symmetry = relative_symmetry_error(
    M_float32,
)

println()
@printf("Relative SPAI residual:    %.6e\n", float32_residual)
@printf("Relative symmetry error:   %.6e\n", float32_symmetry)
println("nnz(M):                    ", nnz(M_float32))
@printf("Density of M:              %.6e\n", nnz(M_float32) / n^2)
@printf("Construction time:         %.4f seconds\n", float32_time)


# Float64 fixed-pattern refinement

println()
println("Float64 fixed-pattern refinement")

refinement_time = @elapsed begin
    M_refined = refine_spai_pattern(
        A,
        M_float32,
    )
end

refined_residual = relative_spai_residual(
    A,
    M_refined,
)

refined_symmetry = relative_symmetry_error(
    M_refined,
)

improvement_factor =
    refined_residual == 0.0 ?
    Inf :
    float32_residual / refined_residual

pattern_valid = pattern_is_subset(
    M_refined,
    M_float32,
)

println()
@printf("Refined residual:          %.6e\n", refined_residual)
@printf("Improvement factor:        %.6e\n", improvement_factor)
@printf("Relative symmetry error:   %.6e\n", refined_symmetry)
println("nnz(M):                    ", nnz(M_refined))
@printf("Density of M:              %.6e\n", nnz(M_refined) / n^2)
println("Pattern valid:             ", pattern_valid)
@printf("Refinement time:           %.4f seconds\n", refinement_time)


# Final summary

println()
println(
    "MSC04515 PCG SUMMARY — " *
    "itmax=$itmax, max_per_column=$max_per_column",
)

@printf(
    "%-24s %-18s %-14s %-18s %-14s\n",
    "Version",
    "Relative residual",
    "nnz(M)",
    "Symmetry error",
    "Time (s)",
)

@printf(
    "%-24s %-18.6e %-14d %-18.6e %-14.4f\n",
    "Float64",
    float64_residual,
    nnz(M_float64),
    float64_symmetry,
    float64_time,
)

@printf(
    "%-24s %-18.6e %-14d %-18.6e %-14.4f\n",
    "Dropped Float32",
    float32_residual,
    nnz(M_float32),
    float32_symmetry,
    float32_time,
)

@printf(
    "%-24s %-18.6e %-14d %-18.6e %-14.4f\n",
    "Refined pattern",
    refined_residual,
    nnz(M_refined),
    refined_symmetry,
    float32_time + refinement_time,
)

println()
@printf(
    "Refinement improvement factor: %.6e\n",
    improvement_factor,
)