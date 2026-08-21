using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf


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


# Constructed a small tri100eigs-style matrix

# Full eigenvalue decompositions are computed, so a smaller versionof tri100eigs4k is used.

Random.seed!(1)

n = 4000
k = 100

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


# Warm-up runs

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

# Compute spectrum of A^{-1} without explicitly forming A^{-1}

println()
println("Computing spectrum of A and A^{-1}...")

eig_A = eigvals(Symmetric(A))

eig_A_inverse = 1.0 ./ eig_A


# Symmetrize approximate inverses

M_full64_sym = Symmetric(
    0.5 .* (M_full64 + transpose(M_full64)),
)

M_lowprec_sym = Symmetric(
    0.5 .* (M_lowprec + transpose(M_lowprec)),
)


# Compute spectra of approximate inverses

println("Computing spectra of the approximate inverses...")

eig_M_full64 = eigvals(M_full64_sym)
eig_M_lowprec = eigvals(M_lowprec_sym)


# Sort spectra

eig_A_inverse_sorted = sort(eig_A_inverse)
eig_M_full64_sorted = sort(eig_M_full64)
eig_M_lowprec_sorted = sort(eig_M_lowprec)


# Relative spectral errors

spectral_error_full64 =
    norm(eig_M_full64_sorted - eig_A_inverse_sorted) /
    norm(eig_A_inverse_sorted)

spectral_error_lowprec =
    norm(eig_M_lowprec_sorted - eig_A_inverse_sorted) /
    norm(eig_A_inverse_sorted)

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


# Summary

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

@printf(
    "%-46s %15.6e\n",
    "Float64 relative spectral error:",
    spectral_error_full64,
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision spectral error:",
    spectral_error_lowprec,
)

@printf(
    "%-46s %15.6e\n",
    "Exact min eigenvalue:",
    minimum(eig_A_inverse_sorted),
)

@printf(
    "%-46s %15.6e\n",
    "Float64 min eigenvalue:",
    minimum(eig_M_full64_sorted),
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision min eigenvalue:",
    minimum(eig_M_lowprec_sorted),
)

@printf(
    "%-46s %15d\n",
    "Float64 negative eigenvalues:",
    count(<(0.0), eig_M_full64_sorted),
)

@printf(
    "%-46s %15d\n",
    "Low-prec negative eigenvalues:",
    count(<(0.0), eig_M_lowprec_sorted),
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


# Plot: spectrum of M versus spectrum of A^{-1}

inverse_spectrum_plot = plot(
    1:n,
    eig_A_inverse_sorted;
    label = "Exact A⁻¹",
    linewidth = 3,
    xlabel = "Ordered eigenvalue index",
    ylabel = "Eigenvalue",
    title = "LOPMR SPAI: spectrum of M compared with A⁻¹",
    legend = :topleft,
    size = (950, 650),
)

plot!(
    inverse_spectrum_plot,
    1:n,
    eig_M_full64_sorted;
    label = "LOPMR Float64 SPAI",
    linewidth = 2,
    linestyle = :dash,
)

plot!(
    inverse_spectrum_plot,
    1:n,
    eig_M_lowprec_sorted;
    label = "LOPMR Float32-SpGEMM SPAI",
    linewidth = 2,
    linestyle = :dot,
)

inverse_output_path = joinpath(
    @__DIR__,
    "tri_small_lopmr_M_vs_Ainverse_spectrum.png",
)

savefig(
    inverse_spectrum_plot,
    inverse_output_path,
)

display(inverse_spectrum_plot)

println()
println("Plot saved to:")
println(inverse_output_path)

# Zoomed spectrum plot: exclude largest eigenvalues

num_excluded = 50
zoom_end = n - num_excluded

zoom_spectral_error_full64 =
    norm(
        eig_M_full64_sorted[1:zoom_end] -
        eig_A_inverse_sorted[1:zoom_end]
    ) /
    norm(eig_A_inverse_sorted[1:zoom_end])

zoom_spectral_error_lowprec =
    norm(
        eig_M_lowprec_sorted[1:zoom_end] -
        eig_A_inverse_sorted[1:zoom_end]
    ) /
    norm(eig_A_inverse_sorted[1:zoom_end])

    @printf(
    "%-46s %15.6e\n",
    "Float64 spectral error excl. top 50:",
    zoom_spectral_error_full64,
)

@printf(
    "%-46s %15.6e\n",
    "Low-prec spectral error excl. top 50:",
    zoom_spectral_error_lowprec,
)

zoomed_spectrum_plot = plot(
    1:zoom_end,
    eig_A_inverse_sorted[1:zoom_end];
    label = "Exact A⁻¹",
    linewidth = 3,
    xlabel = "Ordered eigenvalue index",
    ylabel = "Eigenvalue",
    title = "LOPMR SPAI spectrum (largest 50 eigenvalues excluded)",
    legend = :topleft,
    size = (950, 650),
)

plot!(
    zoomed_spectrum_plot,
    1:zoom_end,
    eig_M_full64_sorted[1:zoom_end];
    label = "LOPMR Float64 SPAI",
    linewidth = 2,
    linestyle = :dash,
)

plot!(
    zoomed_spectrum_plot,
    1:zoom_end,
    eig_M_lowprec_sorted[1:zoom_end];
    label = "LOPMR Float32-SpGEMM SPAI",
    linewidth = 2,
    linestyle = :dot,
)

zoomed_output_path = joinpath(
    @__DIR__,
    "tri4000_lopmr_M_vs_Ainverse_spectrum_zoomed.png",
)

savefig(
    zoomed_spectrum_plot,
    zoomed_output_path,
)

display(zoomed_spectrum_plot)

println("Zoomed plot saved to:")
println(zoomed_output_path)