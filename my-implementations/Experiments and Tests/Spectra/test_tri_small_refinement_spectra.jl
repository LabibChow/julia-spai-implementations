using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
import Measures


# Paths and includes

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

float32_spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)

include(joinpath(utils_dir, "frobenius.jl"))

include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))

include(joinpath(float32_spai_dir, "my_pcg_float32_spai.jl"))
include(joinpath(float32_spai_dir, "my_lopmr_float32_spai.jl"))


# Helper functions

"""
Returns the relative SPAI residual

    ‖I - AM‖F / ‖I‖F

using Float64 arithmetic.
"""
function relative_spai_residual(
    A::AbstractMatrix,
    M::AbstractMatrix,
)
    n = size(A, 1)

    A64 = Matrix{Float64}(A)
    M64 = Matrix{Float64}(M)
    I64 = Matrix{Float64}(I, n, n)

    return frob_norm(I64 - A64 * M64) / sqrt(Float64(n))
end


"""
Returns the relative symmetry error

    ‖M - Mᵀ‖F / ‖M‖F.
"""
function relative_symmetry_error(
    M::AbstractMatrix,
)
    M64 = Matrix{Float64}(M)

    denominator = frob_norm(M64)

    if denominator == 0.0
        return 0.0
    end

    return frob_norm(M64 - M64') / denominator
end


"""
Returns the largest absolute imaginary part among eigenvalues.
"""
function maximum_imaginary_part(values)
    return maximum(abs.(imag.(values)))
end


"""
Returns the largest distance of eigenvalues from the ideal value 1.
"""
function maximum_distance_from_one(values)
    return maximum(abs.(values .- 1.0))
end


"""
Returns the smallest and largest real parts of eigenvalues.
"""
function real_part_range(values)
    real_values = real.(values)

    return minimum(real_values), maximum(real_values)
end


"""
Prints diagnostics for one approximate inverse.
"""
function print_diagnostics(
    label::String,
    A::AbstractMatrix,
    M::AbstractMatrix,
    eigenvalues_M,
    eigenvalues_AM,
)
    relative_residual = relative_spai_residual(
        A,
        M,
    )

    symmetry_error = relative_symmetry_error(
        M,
    )

    minimum_real, maximum_real = real_part_range(
        eigenvalues_AM,
    )

    println()
    println(label)

    @printf(
        "%-42s %15.6e\n",
        "Relative SPAI residual:",
        relative_residual,
    )

    @printf(
        "%-42s %15.6e\n",
        "Relative symmetry error:",
        symmetry_error,
    )

    @printf(
        "%-42s %15.6e\n",
        "Maximum Im(lambda(M)):",
        maximum_imaginary_part(eigenvalues_M),
    )

    @printf(
        "%-42s %15.6e\n",
        "Maximum Im(lambda(AM)):",
        maximum_imaginary_part(eigenvalues_AM),
    )

    @printf(
        "%-42s %15.6e\n",
        "Maximum |lambda(AM)-1|:",
        maximum_distance_from_one(eigenvalues_AM),
    )

    @printf(
        "%-42s [% .6e, % .6e]\n",
        "Real range lambda(AM):",
        minimum_real,
        maximum_real,
    )

    @printf(
        "%-42s %15d\n",
        "nnz(M):",
        nnz(sparse(M)),
    )
end


# Constructed a small tri100eigs-style matrix
# Full spectra are computed, so a small matrix is used.

Random.seed!(1)

n = 200
k = 20

main_diagonal = [
    0.05 .+ rand(k)
    ones(n - k)
]

lower_diagonal = rand(n - 1)

B = spdiagm(
     0 => main_diagonal,
    -1 => lower_diagonal,
)

A_sparse = sparse(
    B * transpose(B),
)

A = Matrix{Float64}(
    A_sparse,
)

println("="^80)
println("Matrix: small tri100eigs-style matrix")
println("Size:   ", size(A_sparse))
println("nnz(A): ", nnz(A_sparse))
println("="^80)


# Initial approximate inverse and Jacobi preconditioner

M0_float64 = spdiagm(
    0 => ones(Float64, n),
)

Pr_float64 = spdiagm(
    0 => 1.0 ./ diag(A_sparse),
)

M0_float32 = SparseMatrixCSC{Float32, Int}(
    M0_float64,
)

Pr_float32 = SparseMatrixCSC{Float32, Int}(
    Pr_float64,
)

A_float32 = SparseMatrixCSC{Float32, Int}(
    A_sparse,
)


# Experiment settings

itmax = 300
tol = 1.0e-15


# Warm-up runs

println()
println("Compiling PCG methods...")

my_pcg(
    A_sparse,
    Pr_float64,
    copy(M0_float64),
    1,
    tol,
)

my_pcg_float32(
    A_float32,
    Pr_float32,
    copy(M0_float32),
    1,
    tol,
)

println()
println("Compiling LOPMR methods...")

my_lopmr(
    A_sparse,
    Pr_float64,
    copy(M0_float64),
    1,
    tol,
)

my_lopmr_float32(
    A_float32,
    Pr_float32,
    copy(M0_float32),
    1,
    tol,
)

GC.gc()


# PCG experiments

println()
println("PCG FLOAT64")

pcg_float64_result = nothing

pcg_float64_time = @elapsed begin
    pcg_float64_result = my_pcg(
        A_sparse,
        Pr_float64,
        copy(M0_float64),
        itmax,
        tol,
    )
end

M_pcg_float64_sparse, R_pcg_float64 = pcg_float64_result


println()
println("PCG FLOAT32")

pcg_float32_result = nothing

pcg_float32_time = @elapsed begin
    pcg_float32_result = my_pcg_float32(
        A_float32,
        Pr_float32,
        copy(M0_float32),
        itmax,
        tol,
    )
end

M_pcg_float32_sparse, R_pcg_float32 = pcg_float32_result


println()
println("PCG FLOAT32 PATTERN REFINED IN FLOAT64")

pcg_refinement_time = @elapsed begin
    global M_pcg_refined_sparse

    M_pcg_refined_sparse = refine_spai_pattern(
        A_sparse,
        M_pcg_float32_sparse,
    )
end


# LOPMR experiments

GC.gc()

println()
println("="^80)
println("LOPMR FLOAT64")
println("="^80)

lopmr_float64_result = nothing

lopmr_float64_time = @elapsed begin
    lopmr_float64_result = my_lopmr(
        A_sparse,
        Pr_float64,
        copy(M0_float64),
        itmax,
        tol,
    )
end

M_lopmr_float64_sparse, R_lopmr_float64 = lopmr_float64_result


println()
println("="^80)
println("LOPMR FLOAT32")
println("="^80)

lopmr_float32_result = nothing

lopmr_float32_time = @elapsed begin
    lopmr_float32_result = my_lopmr_float32(
        A_float32,
        Pr_float32,
        copy(M0_float32),
        itmax,
        tol,
    )
end

M_lopmr_float32_sparse, R_lopmr_float32 = lopmr_float32_result


println()
println("LOPMR FLOAT32 PATTERN REFINED IN FLOAT64")

lopmr_refinement_time = @elapsed begin
    global M_lopmr_refined_sparse

    M_lopmr_refined_sparse = refine_spai_pattern(
        A_sparse,
        M_lopmr_float32_sparse,
    )
end


# Convert matrices to dense Float64

M_pcg_float64 = Matrix{Float64}(
    M_pcg_float64_sparse,
)

M_pcg_float32 = Matrix{Float64}(
    M_pcg_float32_sparse,
)

M_pcg_refined = Matrix{Float64}(
    M_pcg_refined_sparse,
)

M_lopmr_float64 = Matrix{Float64}(
    M_lopmr_float64_sparse,
)

M_lopmr_float32 = Matrix{Float64}(
    M_lopmr_float32_sparse,
)

M_lopmr_refined = Matrix{Float64}(
    M_lopmr_refined_sparse,
)


# Exact inverse

println()
println("Computing exact inverse...")

A_inverse = inv(A)


# Preconditioned matrices

AM_pcg_float64 = A * M_pcg_float64
AM_pcg_float32 = A * M_pcg_float32
AM_pcg_refined = A * M_pcg_refined

AM_lopmr_float64 = A * M_lopmr_float64
AM_lopmr_float32 = A * M_lopmr_float32
AM_lopmr_refined = A * M_lopmr_refined


# Compute spectra

println()
println("Computing eigenvalue spectra...")

eig_A_inverse = eigvals(
    A_inverse,
)

eig_M_pcg_float64 = eigvals(
    M_pcg_float64,
)

eig_M_pcg_float32 = eigvals(
    M_pcg_float32,
)

eig_M_pcg_refined = eigvals(
    M_pcg_refined,
)

eig_AM_pcg_float64 = eigvals(
    AM_pcg_float64,
)

eig_AM_pcg_float32 = eigvals(
    AM_pcg_float32,
)

eig_AM_pcg_refined = eigvals(
    AM_pcg_refined,
)

eig_M_lopmr_float64 = eigvals(
    M_lopmr_float64,
)

eig_M_lopmr_float32 = eigvals(
    M_lopmr_float32,
)

eig_M_lopmr_refined = eigvals(
    M_lopmr_refined,
)

eig_AM_lopmr_float64 = eigvals(
    AM_lopmr_float64,
)

eig_AM_lopmr_float32 = eigvals(
    AM_lopmr_float32,
)

eig_AM_lopmr_refined = eigvals(
    AM_lopmr_refined,
)


# Numerical diagnostics

println()
println("PCG REFINEMENT SPECTRAL SUMMARY")

print_diagnostics(
    "PCG Float64",
    A,
    M_pcg_float64,
    eig_M_pcg_float64,
    eig_AM_pcg_float64,
)

print_diagnostics(
    "PCG Float32",
    A,
    M_pcg_float32,
    eig_M_pcg_float32,
    eig_AM_pcg_float32,
)

print_diagnostics(
    "PCG Float32 pattern refined in Float64",
    A,
    M_pcg_refined,
    eig_M_pcg_refined,
    eig_AM_pcg_refined,
)


println()
println("LOPMR REFINEMENT SPECTRAL SUMMARY")

print_diagnostics(
    "LOPMR Float64",
    A,
    M_lopmr_float64,
    eig_M_lopmr_float64,
    eig_AM_lopmr_float64,
)

print_diagnostics(
    "LOPMR Float32",
    A,
    M_lopmr_float32,
    eig_M_lopmr_float32,
    eig_AM_lopmr_float32,
)

print_diagnostics(
    "LOPMR Float32 pattern refined in Float64",
    A,
    M_lopmr_refined,
    eig_M_lopmr_refined,
    eig_AM_lopmr_refined,
)


# Runtime summary

println()
println("RUNTIME SUMMARY")

@printf(
    "%-48s %15.6f s\n",
    "PCG Float64 construction:",
    pcg_float64_time,
)

@printf(
    "%-48s %15.6f s\n",
    "PCG Float32 construction:",
    pcg_float32_time,
)

@printf(
    "%-48s %15.6f s\n",
    "PCG Float64 refinement:",
    pcg_refinement_time,
)

@printf(
    "%-48s %15.6f s\n",
    "PCG Float32 + refinement total:",
    pcg_float32_time + pcg_refinement_time,
)

@printf(
    "%-48s %15.6f s\n",
    "LOPMR Float64 construction:",
    lopmr_float64_time,
)

@printf(
    "%-48s %15.6f s\n",
    "LOPMR Float32 construction:",
    lopmr_float32_time,
)

@printf(
    "%-48s %15.6f s\n",
    "LOPMR Float64 refinement:",
    lopmr_refinement_time,
)

@printf(
    "%-48s %15.6f s\n",
    "LOPMR Float32 + refinement total:",
    lopmr_float32_time + lopmr_refinement_time,
)


# Plot 1: PCG approximate inverse spectra

pcg_inverse_plot = scatter(
    real.(eig_A_inverse),
    imag.(eig_A_inverse);
    label = "A⁻¹",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "PCG refinement: spectrum of approximate inverses",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    pcg_inverse_plot,
    real.(eig_M_pcg_float64),
    imag.(eig_M_pcg_float64);
    label = "PCG Float64",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    pcg_inverse_plot,
    real.(eig_M_pcg_float32),
    imag.(eig_M_pcg_float32);
    label = "PCG Float32",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    pcg_inverse_plot,
    real.(eig_M_pcg_refined),
    imag.(eig_M_pcg_refined);
    label = "PCG refined",
    markersize = 4,
    markerstrokewidth = 0,
)

hline!(
    pcg_inverse_plot,
    [0.0];
    label = false,
    linestyle = :dot,
)

pcg_inverse_output = joinpath(
    @__DIR__,
    "tri_small_pcg_refinement_inverse_spectra.png",
)

savefig(
    pcg_inverse_plot,
    pcg_inverse_output,
)


# Plot 2: PCG preconditioned spectra

pcg_preconditioned_plot = scatter(
    real.(eig_AM_pcg_float64),
    imag.(eig_AM_pcg_float64);
    label = "A M Float64",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "PCG refinement: spectrum of AM",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    pcg_preconditioned_plot,
    real.(eig_AM_pcg_float32),
    imag.(eig_AM_pcg_float32);
    label = "A M Float32",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    pcg_preconditioned_plot,
    real.(eig_AM_pcg_refined),
    imag.(eig_AM_pcg_refined);
    label = "A M refined",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    pcg_preconditioned_plot,
    [1.0],
    [0.0];
    label = "Ideal eigenvalue",
    markersize = 9,
    markershape = :star5,
)

hline!(
    pcg_preconditioned_plot,
    [0.0];
    label = false,
    linestyle = :dot,
)

pcg_preconditioned_output = joinpath(
    @__DIR__,
    "tri_small_pcg_refinement_preconditioned_spectra.png",
)

savefig(
    pcg_preconditioned_plot,
    pcg_preconditioned_output,
)


# Plot 3: PCG ordered real parts

ordered_pcg_float64 = sort(
    real.(eig_AM_pcg_float64),
)

ordered_pcg_float32 = sort(
    real.(eig_AM_pcg_float32),
)

ordered_pcg_refined = sort(
    real.(eig_AM_pcg_refined),
)

pcg_ordered_plot = plot(
    1:n,
    ordered_pcg_float64;
    label = "A M Float64",
    linewidth = 2,
    xlabel = "Ordered eigenvalue index",
    ylabel = "Real part of eigenvalue",
    title = "PCG refinement: ordered real parts of λ(AM)",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

plot!(
    pcg_ordered_plot,
    1:n,
    ordered_pcg_float32;
    label = "A M Float32",
    linewidth = 2,
    linestyle = :dash,
)

plot!(
    pcg_ordered_plot,
    1:n,
    ordered_pcg_refined;
    label = "A M refined",
    linewidth = 2,
    linestyle = :dot,
)

hline!(
    pcg_ordered_plot,
    [1.0];
    label = "Ideal value",
    linestyle = :dashdot,
    linewidth = 2,
)

pcg_ordered_output = joinpath(
    @__DIR__,
    "tri_small_pcg_refinement_preconditioned_ordered.png",
)

savefig(
    pcg_ordered_plot,
    pcg_ordered_output,
)


# Plot 4: LOPMR approximate inverse spectra

lopmr_inverse_plot = scatter(
    real.(eig_A_inverse),
    imag.(eig_A_inverse);
    label = "A⁻¹",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "LOPMR refinement: spectrum of approximate inverses",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    lopmr_inverse_plot,
    real.(eig_M_lopmr_float64),
    imag.(eig_M_lopmr_float64);
    label = "LOPMR Float64",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    lopmr_inverse_plot,
    real.(eig_M_lopmr_float32),
    imag.(eig_M_lopmr_float32);
    label = "LOPMR Float32",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    lopmr_inverse_plot,
    real.(eig_M_lopmr_refined),
    imag.(eig_M_lopmr_refined);
    label = "LOPMR refined",
    markersize = 4,
    markerstrokewidth = 0,
)

hline!(
    lopmr_inverse_plot,
    [0.0];
    label = false,
    linestyle = :dot,
)

lopmr_inverse_output = joinpath(
    @__DIR__,
    "tri_small_lopmr_refinement_inverse_spectra.png",
)

savefig(
    lopmr_inverse_plot,
    lopmr_inverse_output,
)


# Plot 5: LOPMR preconditioned spectra

lopmr_preconditioned_plot = scatter(
    real.(eig_AM_lopmr_float64),
    imag.(eig_AM_lopmr_float64);
    label = "A M Float64",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "LOPMR refinement: spectrum of AM",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    lopmr_preconditioned_plot,
    real.(eig_AM_lopmr_float32),
    imag.(eig_AM_lopmr_float32);
    label = "A M Float32",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    lopmr_preconditioned_plot,
    real.(eig_AM_lopmr_refined),
    imag.(eig_AM_lopmr_refined);
    label = "A M refined",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    lopmr_preconditioned_plot,
    [1.0],
    [0.0];
    label = "Ideal eigenvalue",
    markersize = 9,
    markershape = :star5,
)

hline!(
    lopmr_preconditioned_plot,
    [0.0];
    label = false,
    linestyle = :dot,
)

lopmr_preconditioned_output = joinpath(
    @__DIR__,
    "tri_small_lopmr_refinement_preconditioned_spectra.png",
)

savefig(
    lopmr_preconditioned_plot,
    lopmr_preconditioned_output,
)


# Plot 6: LOPMR ordered real parts

ordered_lopmr_float64 = sort(
    real.(eig_AM_lopmr_float64),
)

ordered_lopmr_float32 = sort(
    real.(eig_AM_lopmr_float32),
)

ordered_lopmr_refined = sort(
    real.(eig_AM_lopmr_refined),
)

lopmr_ordered_plot = plot(
    1:n,
    ordered_lopmr_float64;
    label = "A M Float64",
    linewidth = 2,
    xlabel = "Ordered eigenvalue index",
    ylabel = "Real part of eigenvalue",
    title = "LOPMR refinement: ordered real parts of λ(AM)",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

plot!(
    lopmr_ordered_plot,
    1:n,
    ordered_lopmr_float32;
    label = "A M Float32",
    linewidth = 2,
    linestyle = :dash,
)

plot!(
    lopmr_ordered_plot,
    1:n,
    ordered_lopmr_refined;
    label = "A M refined",
    linewidth = 2,
    linestyle = :dot,
)

hline!(
    lopmr_ordered_plot,
    [1.0];
    label = "Ideal value",
    linestyle = :dashdot,
    linewidth = 2,
)

lopmr_ordered_output = joinpath(
    @__DIR__,
    "tri_small_lopmr_refinement_preconditioned_ordered.png",
)

savefig(
    lopmr_ordered_plot,
    lopmr_ordered_output,
)


# Display plots

display(pcg_inverse_plot)
display(pcg_preconditioned_plot)
display(pcg_ordered_plot)

display(lopmr_inverse_plot)
display(lopmr_preconditioned_plot)
display(lopmr_ordered_plot)


# Output paths

println()
println("Plots saved to:")
println(pcg_inverse_output)
println(pcg_preconditioned_output)
println(pcg_ordered_output)
println(lopmr_inverse_output)
println(lopmr_preconditioned_output)
println(lopmr_ordered_output)