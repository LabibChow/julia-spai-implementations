using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
using Measures

include(joinpath(@__DIR__, "..", "Utils", "frobenius.jl"))

# Nicolas's reference module
reference_dir = joinpath(@__DIR__, "..", "NicolasReference")

include(joinpath(reference_dir, "MySPAI.jl"))
using .MySPAI

# My implementations
spai_dir = joinpath(@__DIR__, "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))

# Output directory

output_dir = "sanity_check_results"
mkpath(output_dir)

# Generate tri100eigs4k exactly as in Experiment01

Random.seed!(1)

n = 4_000
k = 100

A = spdiagm(
    0  => vcat(0.05 .+ rand(k), ones(n - k)),
    -1 => rand(n - 1)
)

A = sparse(A * A')

itmax = 300
tol = 1e-15

# Jacobi preconditioner
d = diag(A)

if any(iszero, d)
    error("Cannot construct Jacobi preconditioner because A has a zero diagonal entry.")
end

Pr = spdiagm(0 => 1.0 ./ d)

# Initial approximate inverse M₀ = I
M0 = spdiagm(0 => ones(n))

println("Matrix: tri100eigs4k")
println("Size: ", size(A))
println("Nonzeros: ", nnz(A))
println("Maximum iterations: ", itmax)
println("Tolerance: ", tol)
println()

# Run Nicolas's reference implementations

println("Running Nicolas's PMR...")

time_ref_pmr = @elapsed begin
    M_ref_pmr, R_ref_pmr =
        MySPAI.pmr(A, Pr, copy(M0), itmax, tol)
end

println("Finished Nicolas's PMR in $(round(time_ref_pmr, digits=3)) seconds.")
println()

println("Running Nicolas's PCG...")

time_ref_pcg = @elapsed begin
    M_ref_pcg, R_ref_pcg =
        MySPAI.pcg(A, Pr, copy(M0), itmax, tol)
end

println("Finished Nicolas's PCG in $(round(time_ref_pcg, digits=3)) seconds.")
println()

println("Running Nicolas's LOPMR...")

time_ref_lopmr = @elapsed begin
    M_ref_lopmr, R_ref_lopmr =
        MySPAI.lopmr(A, Pr, copy(M0), itmax, tol)
end

println("Finished Nicolas's LOPMR in $(round(time_ref_lopmr, digits=3)) seconds.")
println()

# Run my implementations

println("Running my PMR...")

time_my_pmr = @elapsed begin
    M_my_pmr, R_my_pmr =
        my_pmr(A, Pr, copy(M0), itmax, tol)
end

println("Finished my PMR in $(round(time_my_pmr, digits=3)) seconds.")
println()

println("Running my PCG...")

time_my_pcg = @elapsed begin
    M_my_pcg, R_my_pcg =
        my_pcg(A, Pr, copy(M0), itmax, tol)
end

println("Finished my PCG in $(round(time_my_pcg, digits=3)) seconds.")
println()

println("Running my LOPMR...")

time_my_lopmr = @elapsed begin
    M_my_lopmr, R_my_lopmr =
        my_lopmr(A, Pr, copy(M0), itmax, tol)
end

println("Finished my LOPMR in $(round(time_my_lopmr, digits=3)) seconds.")
println()

# Validate residual histories

function validate_residual_history(name, residuals)
    if isempty(residuals)
        error("$name returned an empty residual history.")
    end

    if any(isnan, residuals)
        @warn "$name residual history contains NaN values."
    end

    if any(isinf, residuals)
        @warn "$name residual history contains Inf values."
    end

    if any(residuals .< 0)
        @warn "$name residual history contains negative values."
    end
end

validate_residual_history("Reference PMR", R_ref_pmr)
validate_residual_history("Reference PCG", R_ref_pcg)
validate_residual_history("Reference LOPMR", R_ref_lopmr)

validate_residual_history("My PMR", R_my_pmr)
validate_residual_history("My PCG", R_my_pcg)
validate_residual_history("My LOPMR", R_my_lopmr)

# Numerical comparison

function compare_residual_histories(name, reference, mine)
    compared_length = min(length(reference), length(mine))

    reference_common = reference[1:compared_length]
    mine_common = mine[1:compared_length]

    absolute_differences =
        abs.(reference_common .- mine_common)

    relative_differences =
        absolute_differences ./ max.(abs.(reference_common), eps(Float64))

    maximum_absolute_difference =
        maximum(absolute_differences)

    maximum_relative_difference =
        maximum(relative_differences)

    final_absolute_difference =
        abs(reference_common[end] - mine_common[end])

    println("--------------------------------------------------")
    println(name)
    println("--------------------------------------------------")
    println("Reference history length: ", length(reference))
    println("My history length:        ", length(mine))
    println("Compared entries:         ", compared_length)
    println()
    println("Initial reference residual: ", reference_common[1])
    println("Initial my residual:        ", mine_common[1])
    println()
    println("Final reference residual:   ", reference_common[end])
    println("Final my residual:          ", mine_common[end])
    println()
    println("Maximum absolute difference: ", maximum_absolute_difference)
    println("Maximum relative difference: ", maximum_relative_difference)
    println("Final absolute difference:   ", final_absolute_difference)
    println()
end

compare_residual_histories("PMR comparison", R_ref_pmr, R_my_pmr)
compare_residual_histories("PCG comparison", R_ref_pcg, R_my_pcg)
compare_residual_histories("LOPMR comparison", R_ref_lopmr, R_my_lopmr)

# Optional comparison of final approximate inverse matrices

function relative_matrix_difference(reference_matrix, my_matrix)
    denominator = sqrt(sum(abs2, reference_matrix))

    numerator =
        sqrt(sum(abs2, reference_matrix - my_matrix))

    return denominator == 0 ? numerator : numerator / denominator
end

println("Final approximate-inverse differences")
println("-------------------------------------")

println(
    "PMR relative matrix difference: ",
    relative_matrix_difference(M_ref_pmr, M_my_pmr)
)

println(
    "PCG relative matrix difference: ",
    relative_matrix_difference(M_ref_pcg, M_my_pcg)
)

println(
    "LOPMR relative matrix difference: ",
    relative_matrix_difference(M_ref_lopmr, M_my_lopmr)
)

println()

# Iteration-axis helper

iteration_axis(residuals) = 0:(length(residuals) - 1)

# Plot 1: My implementations only

plot_my_methods = plot(
    iteration_axis(R_my_pmr),
    R_my_pmr,
    label = "PMR",
    color = :blue,
    linestyle = :dash,
    linewidth = 2,
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "tri100eigs4k — My implementations",
    legend = :bottomleft,
    grid = true,
    size = (850, 500)
)

plot!(
    plot_my_methods,
    iteration_axis(R_my_pcg),
    R_my_pcg,
    label = "PCG",
    color = :red,
    linestyle = :dash,
    linewidth = 2
)

plot!(
    plot_my_methods,
    iteration_axis(R_my_lopmr),
    R_my_lopmr,
    label = "LOPMR",
    color = :green,
    linestyle = :dash,
    linewidth = 2
)

# Plot 2: Reference implementations only

plot_reference = plot(
    iteration_axis(R_ref_pmr),
    R_ref_pmr,
    label = "PMR",
    color = :blue,
    linestyle = :solid,
    linewidth = 3,
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "tri100eigs4k — Nicolas's reference",
    legend = :bottomleft,
    grid = true,
    size = (850, 500)
)

plot!(
    plot_reference,
    iteration_axis(R_ref_pcg),
    R_ref_pcg,
    label = "PCG",
    color = :red,
    linestyle = :solid,
    linewidth = 3
)

plot!(
    plot_reference,
    iteration_axis(R_ref_lopmr),
    R_ref_lopmr,
    label = "LOPMR",
    color = :green,
    linestyle = :solid,
    linewidth = 3
)

# Plot 3: Reference versus my implementations

plot_comparison = plot(
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "tri100eigs4k — Sanity-check comparison",
    legend = :outerright,
    grid = true,
    size = (1200, 700),
    left_margin = 20mm,
    bottom_margin = 14mm,
    right_margin = 10mm,
    dpi = 300
)

# PMR
plot!(
    plot_comparison,
    iteration_axis(R_ref_pmr),
    R_ref_pmr,
    label = "PMR reference",
    color = :blue,
    linestyle = :solid,
    linewidth = 3
)

plot!(
    plot_comparison,
    iteration_axis(R_my_pmr),
    R_my_pmr,
    label = "PMR mine",
    color = :cyan,
    linestyle = :dash,
    linewidth = 2
)

# PCG
plot!(
    plot_comparison,
    iteration_axis(R_ref_pcg),
    R_ref_pcg,
    label = "PCG reference",
    color = :black,
    linestyle = :solid,
    linewidth = 3
)

plot!(
    plot_comparison,
    iteration_axis(R_my_pcg),
    R_my_pcg,
    label = "PCG mine",
    color = :red,
    linestyle = :dash,
    linewidth = 2
)

# LOPMR
plot!(
    plot_comparison,
    iteration_axis(R_ref_lopmr),
    R_ref_lopmr,
    label = "LOPMR reference",
    color = :green,
    linestyle = :solid,
    linewidth = 3
)

plot!(
    plot_comparison,
    iteration_axis(R_my_lopmr),
    R_my_lopmr,
    label = "LOPMR mine",
    color = :yellow,
    linestyle = :dash,
    linewidth = 2
)

display(plot_comparison)

savefig(
    plot_comparison,
    joinpath(output_dir, "tri100eigs4k_reference_vs_mine.png")
)

# Plot 4: Focus on PCG spikes between iterations 100 and 175

plot_pcg_spikes = plot(
    iteration_axis(R_ref_pcg),
    R_ref_pcg,
    label = "PCG reference",
    color = :red,
    linewidth = 3,
    linestyle = :solid,
    yscale = :log10,
    xlims = (80, 190),
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "tri100eigs4k — PCG oscillation region",
    legend = :topright,
    grid = true,
    size = (850, 500)
)

plot!(
    plot_pcg_spikes,
    iteration_axis(R_my_pcg),
    R_my_pcg,
    label = "PCG mine",
    color = :blue,
    linewidth = 2,
    linestyle = :dash
)

display(plot_pcg_spikes)

savefig(
    plot_pcg_spikes,
    joinpath(output_dir, "tri100eigs4k_pcg_spikes.png")
)

# Runtime summary

println()
println("Runtime summary")
println("===============")

@printf("%-20s %12.3f seconds\n", "Reference PMR", time_ref_pmr)
@printf("%-20s %12.3f seconds\n", "My PMR", time_my_pmr)
@printf("%-20s %12.3f seconds\n", "Reference PCG", time_ref_pcg)
@printf("%-20s %12.3f seconds\n", "My PCG", time_my_pcg)
@printf("%-20s %12.3f seconds\n", "Reference LOPMR", time_ref_lopmr)
@printf("%-20s %12.3f seconds\n", "My LOPMR", time_my_lopmr)

println()
println("Figures saved in: ", abspath(output_dir))
println("Sanity check completed.")