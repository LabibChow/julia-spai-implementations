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


"""
Returns the maximum imaginary component among a set of eigenvalues.
"""
function maximum_imaginary_part(values)
    return maximum(abs.(imag.(values)))
end


"""
Returns the maximum distance of the eigenvalues from 1.
"""
function maximum_distance_from_one(values)
    return maximum(abs.(values .- 1.0))
end


# Constructed a small tri100eigs-style matrix

# A full spectrum is computed, so a genuinely small matrix is used.
# This preserves the same matrix construction idea while making dense eigendecompositions practical.

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


# Exact inverse and preconditioned matrices

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
    "%-36s %15.6e\n",
    "Float64 max Im(lambda(M)):",
    maximum_imaginary_part(eig_M_full64),
)

@printf(
    "%-36s %15.6e\n",
    "Low-prec max Im(lambda(M)):",
    maximum_imaginary_part(eig_M_lowprec),
)

@printf(
    "%-36s %15.6e\n",
    "Float64 max |lambda(AM)-1|:",
    maximum_distance_from_one(eig_AM_full64),
)

@printf(
    "%-36s %15.6e\n",
    "Low-prec max |lambda(AM)-1|:",
    maximum_distance_from_one(eig_AM_lowprec),
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


# Plot 1: spectra of A^-1 and computed SPAIs

inverse_spectrum_plot = scatter(
    real.(eig_A_inverse),
    imag.(eig_A_inverse);
    label = "A⁻¹",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "PCG: spectra of A⁻¹ and computed SPAIs",
    legend = :best,
    size = (950, 650),
    margin = 5Measures.mm,
)

scatter!(
    inverse_spectrum_plot,
    real.(eig_M_full64),
    imag.(eig_M_full64);
    label = "PCG Float64 SPAI",
    markersize = 4,
    markerstrokewidth = 0,
)

scatter!(
    inverse_spectrum_plot,
    real.(eig_M_lowprec),
    imag.(eig_M_lowprec);
    label = "PCG Float32-SpGEMM SPAI",
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
    "tri_small_pcg_inverse_spectra.png",
)

savefig(
    inverse_spectrum_plot,
    inverse_output_path,
)


# Plot 2: spectrum of the preconditioned matrices
# A good approximate inverse should make AM close to I, so its eigenvalues should cluster near 1.

preconditioned_spectrum_plot = scatter(
    real.(eig_AM_full64),
    imag.(eig_AM_full64);
    label = "A M Float64",
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title = "Spectrum of the preconditioned matrix AM",
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
    "tri_small_preconditioned_spectra.png",
)

savefig(
    preconditioned_spectrum_plot,
    preconditioned_output_path,
)


# Plot 3: ordered real parts of λ(AM)
# This gives a clearer view when most imaginary parts are tiny.

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
    title = "Ordered real parts of eigenvalues of AM",
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
    "tri_small_preconditioned_spectra_ordered.png",
)

savefig(
    ordered_plot,
    ordered_output_path,
)


# Display and output paths

display(inverse_spectrum_plot)
display(preconditioned_spectrum_plot)
display(ordered_plot)

println()
println("Plots saved to:")
println(inverse_output_path)
println(preconditioned_output_path)
println(ordered_output_path)