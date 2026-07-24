using SparseArrays
using LinearAlgebra
using Random
using Printf
using Plots


# Paths

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

# Frobenius inner product and norm
include(joinpath(utils_dir, "frobenius.jl"))

# Float64 SPAI implementations
include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))

# Float32 SPAI implementations
include(joinpath(float32_dir, "my_pmr_float32_spai.jl"))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl"))
include(joinpath(float32_dir, "my_lopmr_float32_spai.jl"))

# Fixed-pattern Float64 refinement
include(joinpath(spai_dir, "refine_spai_pattern.jl"))


# Helper functions

"""
    relative_spai_residual(A, M)

Computes the relative SPAI residual

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

    return frob_norm(R) / frob_norm(I64)
end


"""
    relative_residual_history(history, n)

Converts an absolute Frobenius residual history into a relative
residual history by dividing by ‖I‖F = √n.
"""
function relative_residual_history(history, n)
    identity_norm = sqrt(Float64(n))

    return Float64.(history) ./ identity_norm
end


"""
    positive_values(values)

Protects logarithmic plots against zero values.
"""
function positive_values(values)
    return max.(Float64.(values), eps(Float64))
end


"""
Checks that the refined matrix contains no nonzero position outside
the original Float32 sparsity pattern.
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


# Float32 plus Float64 refinement experiment

"""
Runs a Float32 SPAI method and then recomputes the best coefficients
in Float64 while keeping the Float32 sparsity pattern fixed.
"""
function test_refinement(
    method_name::String,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)
    println()
    println("="^75)
    println("$method_name: Float32 plus Float64 refinement")
    println("="^75)

    println("Computing Float32 SPAI...")

    float32_time = @elapsed begin
        M32, residual_history = spai_method(
            A,
            Pr,
            M0,
            itmax,
            tol,
        )
    end

    residual_before = relative_spai_residual(A, M32)
    nnz_before = nnz(M32)

    println("Refining fixed sparsity pattern in Float64...")

    refinement_time = @elapsed begin
        M_refined = refine_spai_pattern(A, M32)
    end

    residual_after = relative_spai_residual(A, M_refined)
    nnz_after = nnz(M_refined)

    improvement_factor =
        residual_after == 0.0 ?
        Inf :
        residual_before / residual_after

    pattern_valid = pattern_is_subset(
        M_refined,
        M32,
    )

    println()
    @printf("Float32 residual:          %.6e\n", residual_before)
    @printf("Refined residual:          %.6e\n", residual_after)
    @printf("Improvement factor:        %.6e\n", improvement_factor)

    println()
    println("Float32 nnz(M):            ", nnz_before)
    println("Refined nnz(M):            ", nnz_after)
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
        nnz_before = nnz_before,
        nnz_after = nnz_after,
        pattern_valid = pattern_valid,
        float32_time = float32_time,
        refinement_time = refinement_time,
        total_time = float32_time + refinement_time,
    )
end


# Fully Float64 baseline experiment

"""
Runs one fully Float64 SPAI method.
"""
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
    println("="^75)
    println("$method_name: fully Float64 baseline")
    println("="^75)

    float64_time = @elapsed begin
        M64, residual_history = spai_method(
            A,
            Pr,
            M0,
            itmax,
            tol,
        )
    end

    final_residual = relative_spai_residual(A, M64)

    println()
    @printf("Float64 residual:          %.6e\n", final_residual)
    println("Float64 nnz(M):            ", nnz(M64))
    @printf("Float64 construction time: %.4f seconds\n", float64_time)

    return (
        method = method_name,
        M64 = M64,
        residual_history = residual_history,
        residual = final_residual,
        nnz_M = nnz(M64),
        construction_time = float64_time,
    )
end


# Construct tri100eigs4k

Random.seed!(1)

n = 4000
k = 100

diagonal_entries = vcat(
    0.05 .+ rand(k),
    ones(n - k),
)

subdiagonal_entries = rand(n - 1)

B = spdiagm(
     0 => diagonal_entries,
    -1 => subdiagonal_entries,
)

A = sparse(B * transpose(B))

println("="^75)
println("Matrix: tri100eigs4k")
println("Size:   ", size(A))
println("nnz(A): ", nnz(A))
println("="^75)


# Experiment parameters

itmax = 300
tol = 1.0e-15

A_diagonal = diag(A)

# Float64 input matrices
M0_64 = spdiagm(
    0 => ones(Float64, n),
)

Pr64 = spdiagm(
    0 => 1.0 ./ A_diagonal,
)

# Float32 input matrices
M0_32 = spdiagm(
    0 => ones(Float32, n),
)

Pr32 = spdiagm(
    0 => Float32.(1.0 ./ A_diagonal),
)


# Float64 methods
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


# Float32 plus Float64 refinement experiments

pmr_refined = test_refinement(
    "PMR",
    my_pmr_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
)

pcg_refined = test_refinement(
    "PCG",
    my_pcg_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
)

lopmr_refined = test_refinement(
    "LOPMR",
    my_lopmr_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
)


# Final numerical summary

println()
println("="^115)
println("FINAL RESIDUAL SUMMARY")
println("="^115)

@printf(
    "%-10s %-16s %-16s %-16s %-14s\n",
    "Method",
    "Float64",
    "Float32",
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


println()
println("="^115)
println("TIME AND SPARSITY SUMMARY")
println("="^115)

@printf(
    "%-10s %-14s %-14s %-14s %-14s %-14s %-14s\n",
    "Method",
    "Float64 time",
    "Float32 time",
    "Refine time",
    "Total mixed",
    "Float64 nnz",
    "Refined nnz",
)

for (baseline, refined) in (
    (pmr_float64, pmr_refined),
    (pcg_float64, pcg_refined),
    (lopmr_float64, lopmr_refined),
)
    @printf(
        "%-10s %-14.4f %-14.4f %-14.4f %-14.4f %-14d %-14d\n",
        baseline.method,
        baseline.construction_time,
        refined.float32_time,
        refined.refinement_time,
        refined.total_time,
        baseline.nnz_M,
        refined.nnz_after,
    )
end


# Residual-history plots
# One panel is generated for each method.
#
# Solid line:
#     fully Float64 iteration
#
# Dashed line:
#     Float32 iteration
#
# Final marker:
#     Float32 pattern after Float64 least-squares refinement
#
# The refinement is a postprocessing step, not another SPAI iteration,
# so it is represented as a final point.

function add_method_plot!(
    plot_object,
    subplot_index,
    method_name,
    float64_result,
    refined_result,
    n,
)
    history64 = positive_values(
        relative_residual_history(
            float64_result.residual_history,
            n,
        ),
    )

    history32 = positive_values(
        relative_residual_history(
            refined_result.residual_history,
            n,
        ),
    )

    iterations64 = 0:(length(history64) - 1)
    iterations32 = 0:(length(history32) - 1)

    plot!(
        plot_object,
        iterations64,
        history64;
        subplot = subplot_index,
        label = "Float64",
        linewidth = 2,
    )

    plot!(
        plot_object,
        iterations32,
        history32;
        subplot = subplot_index,
        label = "Float32",
        linewidth = 2,
        linestyle = :dash,
    )

    refined_iteration = length(history32) - 1

    scatter!(
        plot_object,
        [refined_iteration],
        [max(refined_result.residual_after, eps(Float64))];
        subplot = subplot_index,
        label = "Float32 pattern + Float64 refinement",
        markersize = 7,
    )

    title!(
        plot_object,
        "$method_name";
        subplot = subplot_index,
    )

    return nothing
end


comparison_plot = plot(
    layout = (1, 3),
    size = (1500, 500),
    xlabel = "SPAI iteration",
    ylabel = "Relative Frobenius residual",
    yscale = :log10,
    legend = :topright,
    plot_title = "tri100eigs4k: Float64, Float32, and fixed-pattern refinement",
)

add_method_plot!(
    comparison_plot,
    1,
    "PMR",
    pmr_float64,
    pmr_refined,
    n,
)

add_method_plot!(
    comparison_plot,
    2,
    "PCG",
    pcg_float64,
    pcg_refined,
    n,
)

add_method_plot!(
    comparison_plot,
    3,
    "LOPMR",
    lopmr_float64,
    lopmr_refined,
    n,
)

plot_path = joinpath(
    @__DIR__,
    "tri100eigs4k_float64_float32_refinement.png",
)

savefig(comparison_plot, plot_path)
display(comparison_plot)

println()
println("Comparison plot saved to:")
println(plot_path)