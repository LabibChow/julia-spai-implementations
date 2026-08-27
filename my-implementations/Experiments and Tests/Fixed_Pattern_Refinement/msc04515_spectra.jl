using MatrixDepot
using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using Serialization
using Plots
import Measures
# Paths
spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)
float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)
utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)
checkpoint_dir = joinpath(@__DIR__, "msc04515_nodrop_checkpoints",)
# Spectrum output directories
results_dir = joinpath(@__DIR__, "msc04515_nodrop_spectra_results",)
plots_dir = joinpath(@__DIR__, "msc04515_nodrop_spectra_plots",)
mkpath(results_dir)
mkpath(plots_dir)
force_recompute_spectra = false
force_rerun_construction = false
# Includes
include(joinpath(utils_dir, "frobenius.jl",))
# Float64 SPAI implementations
include(joinpath(spai_dir, "my_pmr_spai.jl",))
include(joinpath(spai_dir, "my_pcg_spai.jl",))
include(joinpath(spai_dir, "my_lopmr_spai.jl",))
# Float32 SPAI implementations
include(joinpath(float32_dir, "my_pmr_float32_spai.jl",))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl",))
include(joinpath(float32_dir, "my_lopmr_float32_spai.jl",))
# Float64 fixed-pattern refinement
include(joinpath(spai_dir, "refine_spai_pattern.jl",))
# Relative SPAI residual
function relative_spai_residual(A::SparseMatrixCSC, M::SparseMatrixCSC,)
    n = size(A, 1)
    I64 = spdiagm(0 => ones(Float64, n,),)
    R = I64 - sparse(Float64.(A)) * sparse(Float64.(M))
    return frob_norm(R) / sqrt(Float64(n))
end
# Spectrum statistics
function spectrum_statistics(eigenvalues,)
    λ = Vector{Float64}(eigenvalues,)
    scale = maximum(abs, λ,)
    # Relative tolerance used only for classification.
    # Tiny negative values caused by floating-point roundoff
    # are treated as near zero.
    sign_tol = max(1.0e-12 * scale, eps(Float64),)
    n_positive = count(λi -> λi > sign_tol, λ,)
    n_negative = count(λi -> λi < -sign_tol, λ,)
    n_near_zero = length(λ) - n_positive - n_negative
    return (eigenvalues = λ, sign_tol = sign_tol, n_positive = n_positive, n_negative = n_negative, n_near_zero = n_near_zero,
        minimum = minimum(λ), maximum = maximum(λ),)
end
function compute_M_spectrum(method_name, precision_name, M,)
    println()
    println("Computing spectrum: ", method_name, " — ", precision_name,)
    println("Converting M to dense Float64...")
    M_dense = Matrix{Float64}(M)
    println("Symmetrizing M...")
    M_symmetric = Symmetric(0.5 .* (M_dense + transpose(M_dense)),)
    println("Computing full eigenspectrum...")
    eig_time = @elapsed begin
            λ = eigvals(M_symmetric,)
            global _temporary_spectrum = λ
        end
    λ = _temporary_spectrum
    @printf("Eigendecomposition time: %.4f seconds\n", eig_time,)
    spectrum = spectrum_statistics(λ,)
    @printf("Minimum eigenvalue:       %.6e\n", spectrum.minimum,)
    @printf("Maximum eigenvalue:       %.6e\n", spectrum.maximum,)
    @printf("Sign tolerance:           %.6e\n", spectrum.sign_tol,)
    println("Positive eigenvalues:     ", spectrum.n_positive,)
    println("Negative eigenvalues:     ", spectrum.n_negative,)
    println("Near-zero eigenvalues:    ", spectrum.n_near_zero,)
    return spectrum
end
# Load an already-computed spectrum
function load_spectrum(path, description,)
    println()
    println("SPECTRUM ALREADY EXISTS — LOADING")
    println(description)
    println(path)
    λ = vec(readdlm(path, Float64,),)
    spectrum = spectrum_statistics(λ,)
    @printf("Minimum eigenvalue:       %.6e\n", spectrum.minimum,)
    @printf("Maximum eigenvalue:       %.6e\n", spectrum.maximum,)
    println("Positive eigenvalues:     ", spectrum.n_positive,)
    println("Negative eigenvalues:     ", spectrum.n_negative,)
    println("Near-zero eigenvalues:    ", spectrum.n_near_zero,)
    return spectrum
end
# Existing construction checkpoint paths
function checkpoint_path(method_name, precision_name,)
    slug = lowercase(method_name,)
    precision_slug = lowercase(precision_name,)
    return joinpath(checkpoint_dir, "$(slug)_$(precision_slug)_construction.jls",)
end
# Load checkpoint
function load_checkpoint(path,)
    println()
    println("CONSTRUCTION CHECKPOINT FOUND — LOADING")
    println(path)
    return deserialize(path,)
end
# Obtain construction result
function get_construction(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol,)
    checkpoint = checkpoint_path(method_name, precision_name,)
    # Existing checkpoint
    if !force_rerun_construction && isfile(checkpoint)
        return load_checkpoint(checkpoint,)
    end
    # Checkpoint missing: recompute
    println()
    println("$method_name — $precision_name — NO DROPPING")
    println("WARNING: construction checkpoint was not found.")
    println("The SPAI construction will now be recomputed.")
    raw_result = nothing
    construction_time = @elapsed begin
            raw_result = spai_method(A, Pr, copy(M0), itmax, tol,)
        end
    M, residual_history = raw_result
    result = (method = method_name, precision = precision_name, M = M, residual_history = residual_history, raw_residual =
            residual_history[end], relative_residual = relative_spai_residual(A, M,), nnz_M = nnz(M), construction_time =
            construction_time,)
    serialize(checkpoint, result,)
    println()
    println("New checkpoint saved:")
    println(checkpoint)
    return result
end
# Float64 spectrum
function get_float64_spectrum(method_name, spai_method, A, Pr, M0, itmax, tol,)
    slug = lowercase(method_name,)
    eigenvalue_path = joinpath(results_dir, "$(slug)_float64_eigenvalues.txt",)
    # Spectrum cache exists
    if !force_recompute_spectra && isfile(eigenvalue_path)
        return load_spectrum(eigenvalue_path, "$method_name — Float64",)
    end
    # Load final M from construction checkpoint
    construction = get_construction(method_name, "Float64", spai_method, A, Pr, M0, itmax, tol,)
    println()
    println("Using Float64 construction with nnz(M) = ", nnz(construction.M),)
    # Compute spectrum
    spectrum = compute_M_spectrum(method_name, "Float64", construction.M,)
    writedlm(eigenvalue_path, spectrum.eigenvalues,)
    println()
    println("Eigenvalues saved to:")
    println(eigenvalue_path)
    construction = nothing
    GC.gc()
    return spectrum
end
# Float32 + Float64 refinement spectrum
function get_refined_spectrum(method_name, spai_method, A, Pr, M0, itmax, tol,)
    slug = lowercase(method_name,)
    eigenvalue_path = joinpath(results_dir, "$(slug)_float32_refined_eigenvalues.txt",)
    # Refined spectrum cache exists
    if !force_recompute_spectra && isfile(eigenvalue_path)
        return load_spectrum(eigenvalue_path, "$method_name — Float32 + refinement",)
    end
    # Load saved Float32 construction
    construction = get_construction(method_name, "Float32", spai_method, A, Pr, M0, itmax, tol,)
    println()
    println("Refining ", method_name, " Float32 pattern in Float64...")
    # Fixed-pattern refinement
    M_refined = nothing
    refinement_time = @elapsed begin
            M_refined = refine_spai_pattern(A, construction.M,)
        end
    @printf("Refinement time: %.4f seconds\n", refinement_time,)
    println("Refined nnz(M): ", nnz(M_refined),)
    # Safety check
    if !all(isfinite, nonzeros(M_refined),)
        error("$method_name Float32 refined matrix " * "contains Inf or NaN.")
    end
    # Spectrum of refined M
    spectrum = compute_M_spectrum(method_name, "Float32 + refinement", M_refined,)
    writedlm(eigenvalue_path, spectrum.eigenvalues,)
    println()
    println("Eigenvalues saved to:")
    println(eigenvalue_path)
    construction = nothing
    M_refined = nothing
    GC.gc()
    return spectrum
end
# Spectrum of A^{-1}
function get_Ainverse_spectrum(A,)
    eigenvalue_path = joinpath(results_dir, "A_inverse_eigenvalues.txt",)
    # Cached spectrum
    if !force_recompute_spectra && isfile(eigenvalue_path)
        println()
        println("A^-1 SPECTRUM ALREADY EXISTS — LOADING")
        println(eigenvalue_path)
        return vec(readdlm(eigenvalue_path, Float64,),)
    end
    # Compute spectrum of A
    println()
    println("Computing exact spectrum of A^-1")
    A_dense = Matrix{Float64}(A)
    λ_A = eigvals(Symmetric(A_dense,),)
    @printf("Minimum eigenvalue of A: %.6e\n", minimum(λ_A),)
    @printf("Maximum eigenvalue of A: %.6e\n", maximum(λ_A),)
    if minimum(λ_A) <= 0.0
        error("Expected msc04515 to be SPD, but a " * "non-positive eigenvalue of A was detected.")
    end
    λ_A_inverse = 1.0 ./ λ_A
    writedlm(eigenvalue_path, λ_A_inverse,)
    @printf("A^-1 minimum eigenvalue: %.6e\n", minimum(λ_A_inverse),)
    @printf("A^-1 maximum eigenvalue: %.6e\n", maximum(λ_A_inverse),)
    return λ_A_inverse
end
# Spectrum rug helper
function rug_coordinates(values, y; halfheight = 0.08,)
    xs = Float64[]
    ys = Float64[]
    for x in values
        push!(xs, abs(x),)
        push!(xs, abs(x),)
        push!(xs, NaN,)
        push!(ys, y - halfheight,)
        push!(ys, y + halfheight,)
        push!(ys, NaN,)
    end
    return xs, ys
end
# Add one SPAI spectrum row
function add_spectrum_row!(p, spectrum, row, subplot_index,)
    λ = spectrum.eigenvalues
    tol_local = spectrum.sign_tol
    positive = λ[λ .> tol_local]
    negative = λ[λ .< -tol_local]
    offset = 0.14
    # Positive eigenvalues ABOVE
    if !isempty(positive,)
        x, y = rug_coordinates(positive, row + offset,)
        plot!(p, x, y; subplot = subplot_index, color = :black, linewidth = 0.7, alpha = 0.8, label = false,)
    end
    # Negative eigenvalues BELOW
    if !isempty(negative,)
        x, y = rug_coordinates(negative, row - offset,)
        plot!(p, x, y; subplot = subplot_index, color = :red, linewidth = 0.7, alpha = 0.8, label = false,)
    end
    return nothing
end
# A^{-1} row
function add_Ainverse_row!(p, λ_A_inverse, row, subplot_index,)
    offset = 0.14
    x, y = rug_coordinates(λ_A_inverse, row + offset,)
    plot!(p, x, y; subplot = subplot_index, color = :black, linewidth = 0.7, alpha = 0.8, label = false,)
    return nothing
end
# Extract meaningful eigenvalue magnitudes
function meaningful_magnitudes(spectrum,)
    λ = spectrum.eigenvalues
    meaningful = λ[abs.(λ) .> spectrum.sign_tol]
    return abs.(meaningful,)
end
# Load msc04515
println()
println("LOADING MSC04515")
md = mdopen("Boeing/msc04515",)
A = sparse(Float64.(md.A),)
n = size(A, 1,)
println()
println("Matrix: msc04515")
println("Size:   ", size(A),)
println("nnz(A): ", nnz(A),)
# Experiment parameters
itmax = 5000
tol = 1.0e-15
println()
println("Experiment parameters:")
println("itmax:    ", itmax,)
println("tol:      ", tol,)
println("dropping: NONE",)
println("construction checkpoint reuse: ", !force_rerun_construction,)
println("spectrum cache reuse: ", !force_recompute_spectra,)
# Initial matrices
A_diagonal = diag(A)
if any(iszero, A_diagonal,)
    error("Cannot construct the Jacobi preconditioner " * "because A contains a zero diagonal entry.")
end
# Float64
M0_64 = spdiagm(0 => ones(Float64, n,),)
Pr64 = spdiagm(0 => 1.0 ./ A_diagonal,)
# Float32
M0_32 = spdiagm(0 => ones(Float32, n,),)
Pr32 = spdiagm(0 => Float32.(1.0 ./ A_diagonal),)
# A^{-1} reference spectrum
λ_A_inverse = get_Ainverse_spectrum(A,)
GC.gc()
# FLOAT64
pmr64 = get_float64_spectrum("PMR", my_pmr, A, Pr64, M0_64, itmax, tol,)
GC.gc()
lopmr64 = get_float64_spectrum("LOPMR", my_lopmr, A, Pr64, M0_64, itmax, tol,)
GC.gc()
pcg64 = get_float64_spectrum("PCG", my_pcg, A, Pr64, M0_64, itmax, tol,)
GC.gc()
# FLOAT32 + FLOAT64 FIXED-PATTERN REFINEMENT
pmr32 = get_refined_spectrum("PMR", my_pmr_float32, A, Pr32, M0_32, itmax, tol,)
GC.gc()
lopmr32 = get_refined_spectrum("LOPMR", my_lopmr_float32, A, Pr32, M0_32, itmax, tol,)
GC.gc()
pcg32 = get_refined_spectrum("PCG", my_pcg_float32, A, Pr32, M0_32, itmax, tol,)
GC.gc()
# Save spectrum summary
summary_path = joinpath(results_dir, "msc04515_spectrum_summary.txt",)
open(summary_path, "w",) do io
    println(io, "msc04515 — no-dropping spectra",)
    println(io, "Float64 vs Float32 + Float64 fixed-pattern refinement",)
    println(io, "itmax = $itmax",)
    println(io, "tol = $tol",)
    println(io, "dropping = none",)
    println(io)
    @printf(io, "%-8s %-24s %8s %8s %8s %16s %16s\n", "Method", "Precision", "Pos", "Neg", "Zero", "lambda_min", "lambda_max",)
    for (method, precision, spectrum,) in (("PMR", "Float64", pmr64,), ("LOPMR", "Float64", lopmr64,), ("PCG", "Float64",
            pcg64,), ("PMR", "F32 + refinement", pmr32,), ("LOPMR", "F32 + refinement", lopmr32,), ("PCG",
            "F32 + refinement", pcg32,),)
        @printf(io, "%-8s %-24s %8d %8d %8d %16.6e %16.6e\n", method, precision, spectrum.n_positive, spectrum.n_negative,
            spectrum.n_near_zero, spectrum.minimum, spectrum.maximum,)
    end
    println(io)
    @printf(io, "A^-1 minimum eigenvalue: %.16e\n", minimum(λ_A_inverse),)
    @printf(io, "A^-1 maximum eigenvalue: %.16e\n", maximum(λ_A_inverse),)
end
# Common logarithmic x-axis range
# BOTH panels use exactly the same scale.
all_magnitudes = vcat(abs.(λ_A_inverse), meaningful_magnitudes(pmr64,), meaningful_magnitudes(lopmr64,),
        meaningful_magnitudes(pcg64,), meaningful_magnitudes(pmr32,), meaningful_magnitudes(lopmr32,),
        meaningful_magnitudes(pcg32,),)
# Remove anything unusable on a logarithmic axis.
all_magnitudes = all_magnitudes[(isfinite.(all_magnitudes)) .& (all_magnitudes .> 0.0)]
xmin_raw = minimum(all_magnitudes,)
xmax_raw = maximum(all_magnitudes,)
# Round outward to full decades.
xmin = 10.0^floor(log10(xmin_raw,),)
xmax = 10.0^ceil(log10(xmax_raw,),)
println()
println("COMMON SPECTRAL X-AXIS")
println("xmin = ", xmin,)
println("xmax = ", xmax,)
# TWO-PANEL SPECTRUM FIGURE
spectra_plot = plot(layout = (2, 1), size = (950, 800), dpi = 300, plot_title = "msc04515", plot_titlefontsize = 18,
        plot_titlevspan = 0.05, left_margin = 18 * Measures.mm, right_margin = 8 * Measures.mm, bottom_margin = 10 * Measures.mm,
        top_margin = 8 * Measures.mm,)
panel_titles = ("Float64", "Float32 + refinement",)
panel_spectra = ((pmr64, lopmr64, pcg64,), (pmr32, lopmr32, pcg32,),)
# Draw panels
for subplot_index in 1:2
    plot!(spectra_plot; subplot = subplot_index, xscale = :log10, xlims = (xmin, xmax,), ylims = (0.5, 4.5,), yticks = (
                [4, 3, 2, 1], ["PMR", "LOPMR", "PCG", "A⁻¹",],), xlabel = "Eigenvalue magnitude", title = panel_titles[
                subplot_index], titlefontsize = 14, legend = false, grid = true, framestyle = :box,)
    pmr_spectrum, lopmr_spectrum, pcg_spectrum = panel_spectra[subplot_index]
    # PMR
    add_spectrum_row!(spectra_plot, pmr_spectrum, 4, subplot_index,)
    # LOPMR
    add_spectrum_row!(spectra_plot, lopmr_spectrum, 3, subplot_index,)
    # PCG
    add_spectrum_row!(spectra_plot, pcg_spectrum, 2, subplot_index,)
    # A^{-1}
    add_Ainverse_row!(spectra_plot, λ_A_inverse, 1, subplot_index,)
end
# Legend
plot!(spectra_plot, [NaN], [NaN]; subplot = 1, color = :black, linewidth = 2, label = "Positive eigenvalues",)
plot!(spectra_plot, [NaN], [NaN]; subplot = 1, color = :red, linewidth = 2, label = "Negative eigenvalues",)
plot!(spectra_plot; subplot = 1, legend = :outertop, legend_column = -1,)
# Save final figure
plot_path = joinpath(plots_dir, "msc04515_float64_float32_refined_spectra.png",)
savefig(spectra_plot, plot_path,)
display(spectra_plot,)
# Finished
println()
println("MSC04515 SPECTRUM EXPERIMENT COMPLETE")
println()
println("Spectrum data:")
println(abspath(results_dir))
println()
println("Combined spectrum figure:")
println(abspath(plot_path))
println()
println("Spectrum summary:")
println(abspath(summary_path))
