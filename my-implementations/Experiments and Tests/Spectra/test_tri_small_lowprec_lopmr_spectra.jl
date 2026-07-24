using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
import Measures


# Paths and includes

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

include(joinpath(utils_dir, "frobenius.jl"))

include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_lowprec_spgemm.jl"))


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
Returns the largest absolute imaginary part among the eigenvalues.
"""
function maximum_imaginary_part(values)
    return maximum(abs.(imag.(values)))
end


"""
Returns the largest distance of the eigenvalues from the ideal value 1.
"""
function maximum_distance_from_one(values)
    return maximum(abs.(values .- 1.0))
end


"""
Returns the smallest and largest real parts of the eigenvalues.
"""
function real_part_range(values)
    real_values = real.(values)

    return minimum(real_values), maximum(real_values)
end


# Constructed a small tri100eigs-style matrix

# Full eigenvalue decompositions are computed, so a smaller versionof tri100eigs4k is used.

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

A_sparse = sparse(B * transpose(B))
A = Matrix{Float64}(A_sparse)

println("Matrix: small tri100eigs-style matrix")
println("Size:   ", size(A))
println("nnz(A): ", nnz(A_sparse))


# Initial approximate inverse and Jacobi preconditioner

M0_sparse = spdiagm(
    0 => ones(Float64, n),
)

Pr_sparse = spdiagm(
    0 => 1.0 ./ diag(A),
)


# Experiment settings

itmax = 300
tol = 1.0e-15


# ============================================================
# Warm-up runs
# ============================================================

println()
println("Compiling full Float64 LOPMR...")

my_lopmr(
    A_sparse,
    Pr_sparse,
    copy(M0_sparse),
    1,
    tol,
)

println()
println("Compiling LOPMR with Float32 SpGEMMs...")

my_lopmr_lowprec_spgemm(
    A_sparse,
    Pr_sparse,
    copy(M0_sparse),
    1,
    tol,
)

GC.gc()


# Full Float64 LOPMR

println()
println("FULL FLOAT64 LOPMR")

full64_result = nothing

full64_time = @elapsed begin
    full64_result = my_lopmr(
        A_sparse,
        Pr_sparse,
        copy(M0_sparse),
        itmax,
        tol,
    )
end

M_full64_sparse, R_full64 = full64_result

M_full64 = Matrix{Float64}(M_full64_sparse)


# LOPMR with Float32 A*Z, A*P and A*M

GC.gc()

println()
println("LOPMR WITH FLOAT32 A*Z, A*P AND A*M")

lowprec_result = nothing

lowprec_time = @elapsed begin
    lowprec_result = my_lopmr_lowprec_spgemm(
        A_sparse,
        Pr_sparse,
        copy(M0_sparse),
        itmax,
        tol,
    )
end

M_lowprec_sparse, R_lowprec, timings = lowprec_result

M_lowprec = Matrix{Float64}(M_lowprec_sparse)


# Exact inverse and preconditioned matrices

println()
println("Computing exact inverse and preconditioned matrices...")

A_inverse = inv(A)

AM_full64 = A * M_full64
AM_lowprec = A * M_lowprec


# Compute full spectra

println()
println("Computing full eigenvalue spectra...")

eig_A_inverse = eigvals(A_inverse)

eig_M_full64 = eigvals(M_full64)
eig_M_lowprec = eigvals(M_lowprec)

eig_AM_full64 = eigvals(AM_full64)
eig_AM_lowprec = eigvals(AM_lowprec)


# Numerical diagnostics

relative_residual_full64 = relative_spai_residual(
    A,
    M_full64,
)

relative_residual_lowprec = relative_spai_residual(
    A,
    M_lowprec,
)

symmetry_error_full64 = relative_symmetry_error(
    M_full64,
)

symmetry_error_lowprec = relative_symmetry_error(
    M_lowprec,
)

full64_min_real, full64_max_real = real_part_range(
    eig_AM_full64,
)

lowprec_min_real, lowprec_max_real = real_part_range(
    eig_AM_lowprec,
)


# Summary
# ============================================================

println()
println("LOPMR SPECTRAL AND SPAI SUMMARY")

@printf(
    "%-46s %15.6e\n",
    "Float64 relative SPAI residual:",
    relative_residual_full64,
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision relative SPAI residual:",
    relative_residual_lowprec,
)

@printf(
    "%-46s %15.6e\n",
    "Float64 symmetry error:",
    symmetry_error_full64,
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision symmetry error:",
    symmetry_error_lowprec,
)

@printf(
    "%-46s %15.6e\n",
    "Float64 max Im(lambda(M)):",
    maximum_imaginary_part(eig_M_full64),
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision max Im(lambda(M)):",
    maximum_imaginary_part(eig_M_lowprec),
)

@printf(
    "%-46s %15.6e\n",
    "Float64 max Im(lambda(AM)):",
    maximum_imaginary_part(eig_AM_full64),
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision max Im(lambda(AM)):",
    maximum_imaginary_part(eig_AM_lowprec),
)

@printf(
    "%-46s %15.6e\n",
    "Float64 max |lambda(AM)-1|:",
    maximum_distance_from_one(eig_AM_full64),
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision max |lambda(AM)-1|:",
    maximum_distance_from_one(eig_AM_lowprec),
)

@printf(
    "%-46s [% .6e, % .6e]\n",
    "Float64 real range lambda(AM):",
    full64_min_real,
    full64_max_real,
)

@printf(
    "%-46s [% .6e, % .6e]\n",
    "Low-precision real range lambda(AM):",
    lowprec_min_real,
    lowprec_max_real,
)

@printf(
    "%-46s %15.6f s\n",
    "Float64 construction time:",
    full64_time,
)

@printf(
    "%-46s %15.6f s\n",
    "Low-precision construction time:",
    lowprec_time,
)

@printf(
    "%-46s %15.6e\n",
    "Float64 nnz(M)/n^2:",
    nnz(M_full64_sparse) / n^2,
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision nnz(M)/n^2:",
    nnz(M_lowprec_sparse) / n^2,
)


# Timing breakdown

println()
println("Low-precision timing breakdown")
println("-----")

@printf(
    "%-46s %15.6f s\n",
    "Z -> Float32 conversion:",
    timings.Z_to_Float32,
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*Z SpGEMM:",
    timings.AZ_Float32_SpGEMM,
)

@printf(
    "%-46s %15.6f s\n",
    "A*Z -> Float64 conversion:",
    timings.AZ_to_Float64,
)

@printf(
    "%-46s %15.6f s\n",
    "P -> Float32 conversion:",
    timings.P_to_Float32,
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*P SpGEMM:",
    timings.AP_Float32_SpGEMM,
)

@printf(
    "%-46s %15.6f s\n",
    "A*P -> Float64 conversion:",
    timings.AP_to_Float64,
)

@printf(
    "%-46s %15.6f s\n",
    "M -> Float32 conversion:",
    timings.M_to_Float32,
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*M SpGEMM:",
    timings.AM_Float32_SpGEMM,
)

@printf(
    "%-46s %15.6f s\n",
    "A*M -> Float64 conversion:",
    timings.AM_to_Float64,
)

@printf(
    "%-46s %15.6f s\n",
    "Float64 Pr*AZ SpGEMM:",
    timings.PrAZ_Float64_SpGEMM,
)

@printf(
    "%-46s %15.6f s\n",
    "Float64 Pr*AP SpGEMM:",
    timings.PrAP_Float64_SpGEMM,
)

@printf(
    "%-46s %15.6f s\n",
    "Total conversion time:",
    timings.total_conversion,
)

@printf(
    "%-46s %15.6f s\n",
    "Total Float32 SpGEMM time:",
    timings.total_Float32_SpGEMM,
)

@printf(
    "%-46s %15.6f s\n",
    "Total preconditioner SpGEMM time:",
    timings.total_preconditioner_SpGEMM,
)


# Plot 1: spectra of A^-1 and computed SPAIs

inverse_spectrum_plot = scatter(
    real.(eig_A_inverse),
    imag.(eig_A_inverse);
    label = "A⁻¹",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "LOPMR: spectra of A⁻¹ and computed SPAIs",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    inverse_spectrum_plot,
    real.(eig_M_full64),
    imag.(eig_M_full64);
    label = "LOPMR Float64 SPAI",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    inverse_spectrum_plot,
    real.(eig_M_lowprec),
    imag.(eig_M_lowprec);
    label = "LOPMR Float32-SpGEMM SPAI",
    markersize = 4,
    markerstrokewidth = 0,
)

hline!(
    inverse_spectrum_plot,
    [0.0];
    label = false,
    linestyle = :dot,
)

inverse_output_path = joinpath(
    @__DIR__,
    "tri_small_lopmr_inverse_spectra.png",
)

savefig(
    inverse_spectrum_plot,
    inverse_output_path,
)


# Plot 2: spectrum of preconditioned matrices AM

preconditioned_spectrum_plot = scatter(
    real.(eig_AM_full64),
    imag.(eig_AM_full64);
    label = "A M Float64",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "LOPMR: spectrum of the preconditioned matrix AM",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    preconditioned_spectrum_plot,
    real.(eig_AM_lowprec),
    imag.(eig_AM_lowprec);
    label = "A M Float32 SpGEMMs",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    preconditioned_spectrum_plot,
    [1.0],
    [0.0];
    label = "Ideal eigenvalue",
    markersize = 9,
    markershape = :star5,
)

hline!(
    preconditioned_spectrum_plot,
    [0.0];
    label = false,
    linestyle = :dot,
)

preconditioned_output_path = joinpath(
    @__DIR__,
    "tri_small_lopmr_preconditioned_spectra.png",
)

savefig(
    preconditioned_spectrum_plot,
    preconditioned_output_path,
)


# Plot 3: ordered real parts of lambda(AM)

ordered_full64 = sort(
    real.(eig_AM_full64),
)

ordered_lowprec = sort(
    real.(eig_AM_lowprec),
)

ordered_plot = plot(
    1:n,
    ordered_full64;
    label = "A M Float64",
    linewidth = 2,
    xlabel = "Ordered eigenvalue index",
    ylabel = "Real part of eigenvalue",
    title = "LOPMR: ordered real parts of eigenvalues of AM",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

plot!(
    ordered_plot,
    1:n,
    ordered_lowprec;
    label = "A M Float32 SpGEMMs",
    linewidth = 2,
    linestyle = :dash,
)

hline!(
    ordered_plot,
    [1.0];
    label = "Ideal value",
    linestyle = :dot,
    linewidth = 2,
)

ordered_output_path = joinpath(
    @__DIR__,
    "tri_small_lopmr_preconditioned_spectra_ordered.png",
)

savefig(
    ordered_plot,
    ordered_output_path,
)


# Display plots and print paths

display(inverse_spectrum_plot)
display(preconditioned_spectrum_plot)
display(ordered_plot)

println()
println("Plots saved to:")
println(inverse_output_path)
println(preconditioned_output_path)
println(ordered_output_path)