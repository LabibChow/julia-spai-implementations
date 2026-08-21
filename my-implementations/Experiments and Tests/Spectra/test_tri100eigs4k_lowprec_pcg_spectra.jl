using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf


# Paths and includes

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

include(joinpath(utils_dir, "frobenius.jl"))

include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_pcg_lowprec_spgemm.jl"))


# Helper functions

"""
Returns the relative SPAI residual ‖I-AM‖F / ‖I‖F.
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
Returns the relative symmetry error ‖M-Mᵀ‖F / ‖M‖F.
"""
function relative_symmetry_error(M::AbstractMatrix)
    M64 = Matrix{Float64}(M)
    denominator = frob_norm(M64)

    denominator == 0.0 && return 0.0

    return frob_norm(M64 - M64') / denominator
end


# Constructed a small tri100eigs-style matrix

# A full spectrum is computed, so a genuinely small matrix is used.
# This preserves the same matrix construction idea while making dense eigendecompositions practical.

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

M0 = Matrix{Float64}(I, n, n)

Pr = Diagonal(
    1.0 ./ diag(A),
)


# Experiment settings

itmax = 300
tol = 1.0e-15


# Warm-up

println()
println("Compiling full Float64 PCG...")
my_pcg(
    sparse(A),
    sparse(Pr),
    sparse(M0),
    1,
    tol,
)

println("Compiling low-precision-SpGEMM PCG...")
my_pcg_lowprec_spgemm(
    sparse(A),
    sparse(Pr),
    sparse(M0),
    1,
    tol,
)

GC.gc()


# Full Float64 PCG SPAI

println()
println("FULL FLOAT64 PCG")

full64_time = @elapsed begin
    global M_full64_sparse, R_full64

    M_full64_sparse, R_full64 = my_pcg(
        sparse(A),
        sparse(Pr),
        sparse(M0),
        itmax,
        tol,
    )
end

M_full64 = Matrix{Float64}(M_full64_sparse)


# PCG with Float32 A*P and A*M

GC.gc()

println()
println("PCG WITH FLOAT32 SpGEMMs")

lowprec_time = @elapsed begin
    global M_lowprec_sparse
    global R_lowprec
    global lowprec_timings

    M_lowprec_sparse,
    R_lowprec,
    lowprec_timings = my_pcg_lowprec_spgemm(
        sparse(A),
        sparse(Pr),
        sparse(M0),
        itmax,
        tol,
    )
end

M_lowprec = Matrix{Float64}(M_lowprec_sparse)


# Compute spectrum of A^{-1} without explicitly forming A^{-1}

println()
println("Computing spectrum of A and A^{-1}...")

# A is symmetric positive definite, so used Symmetric
eig_A = eigvals(Symmetric(A))

# If λ is an eigenvalue of A, then 1/λ is an eigenvalue of A^{-1}
eig_A_inverse = 1.0 ./ eig_A


# Symmetrize the approximate inverses

M_full64_sym = Symmetric(
    0.5 .* (M_full64 + transpose(M_full64)),
)

M_lowprec_sym = Symmetric(
    0.5 .* (M_lowprec + transpose(M_lowprec)),
)


# Compute spectra of the approximate inverses

println("Computing spectra of the approximate inverses...")

eig_M_full64 = eigvals(M_full64_sym)
eig_M_lowprec = eigvals(M_lowprec_sym)


# Sort spectra for direct comparison

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

residual_full64 = relative_spai_residual(
    A,
    M_full64,
)

residual_lowprec = relative_spai_residual(
    A,
    M_lowprec,
)

symmetry_full64 = relative_symmetry_error(
    M_full64,
)

symmetry_lowprec = relative_symmetry_error(
    M_lowprec,
)

println()
println("="^85)
println("SPECTRAL AND SPAI SUMMARY")
println("="^85)

@printf(
    "%-36s %15.6e\n",
    "Float64 relative SPAI residual:",
    residual_full64,
)

@printf(
    "%-36s %15.6e\n",
    "Low-precision relative residual:",
    residual_lowprec,
)

@printf(
    "%-36s %15.6e\n",
    "Float64 symmetry error:",
    symmetry_full64,
)

@printf(
    "%-36s %15.6e\n",
    "Low-precision symmetry error:",
    symmetry_lowprec,
)

@printf(
    "%-36s %15.6f s\n",
    "Float64 construction time:",
    full64_time,
)

@printf(
    "%-36s %15.6f s\n",
    "Low-precision construction time:",
    lowprec_time,
)

@printf(
    "%-36s %15.6e\n",
    "Float64 relative spectral error:",
    spectral_error_full64,
)

@printf(
    "%-36s %15.6e\n",
    "Low-precision spectral error:",
    spectral_error_lowprec,
)

@printf(
    "%-36s %15.6e\n",
    "Exact min eigenvalue:",
    minimum(eig_A_inverse_sorted),
)

@printf(
    "%-36s %15.6e\n",
    "Float64 min eigenvalue:",
    minimum(eig_M_full64_sorted),
)

@printf(
    "%-36s %15.6e\n",
    "Low-precision min eigenvalue:",
    minimum(eig_M_lowprec_sorted),
)

@printf(
    "%-36s %15d\n",
    "Float64 negative eigenvalues:",
    count(<(0.0), eig_M_full64_sorted),
)

@printf(
    "%-36s %15d\n",
    "Low-prec negative eigenvalues:",
    count(<(0.0), eig_M_lowprec_sorted),
)

@printf(
    "%-36s %15.6e\n",
    "Float64 nnz(M)/n^2:",
    nnz(M_full64_sparse) / n^2,
)

@printf(
    "%-36s %15.6e\n",
    "Low-precision nnz(M)/n^2:",
    nnz(M_lowprec_sparse) / n^2,
)

# Plot: spectrum of M versus spectrum of A^{-1}

inverse_spectrum_plot = plot(
    1:n,
    eig_A_inverse_sorted;
    label = "Exact A⁻¹",
    linewidth = 3,
    xlabel = "Ordered eigenvalue index",
    ylabel = "Eigenvalue",
    title = "PCG SPAI: spectrum of M compared with A⁻¹",
    legend = :topleft,
    size = (950, 650),
)

plot!(
    inverse_spectrum_plot,
    1:n,
    eig_M_full64_sorted;
    label = "PCG Float64 SPAI",
    linewidth = 2,
    linestyle = :dash,
)

plot!(
    inverse_spectrum_plot,
    1:n,
    eig_M_lowprec_sorted;
    label = "PCG Float32-SpGEMM SPAI",
    linewidth = 2,
    linestyle = :dot,
)

inverse_output_path = joinpath(
    @__DIR__,
    "tri_small_pcg_M_vs_Ainverse_spectrum.png",
)

savefig(
    inverse_spectrum_plot,
    inverse_output_path,
)

display(inverse_spectrum_plot)

println()
println("Plot saved to:")
println(inverse_output_path)

# Zoomed spectrum plot: excluded largest 50 eigenvalues

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
    title = "PCG SPAI spectrum (largest 50 eigenvalues excluded)",
    legend = :topleft,
    size = (950, 650),
)

plot!(
    zoomed_spectrum_plot,
    1:zoom_end,
    eig_M_full64_sorted[1:zoom_end];
    label = "PCG Float64 SPAI",
    linewidth = 2,
    linestyle = :dash,
)

plot!(
    zoomed_spectrum_plot,
    1:zoom_end,
    eig_M_lowprec_sorted[1:zoom_end];
    label = "PCG Float32-SpGEMM SPAI",
    linewidth = 2,
    linestyle = :dot,
)

zoomed_output_path = joinpath(
    @__DIR__,
    "tri4000_pcg_M_vs_Ainverse_spectrum_zoomed.png",
)

savefig(
    zoomed_spectrum_plot,
    zoomed_output_path,
)

display(zoomed_spectrum_plot)

println()
println("Zoomed plot saved to:")
println(zoomed_output_path)