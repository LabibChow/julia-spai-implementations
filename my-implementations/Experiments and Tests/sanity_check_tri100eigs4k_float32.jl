using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
import Measures

include(joinpath(@__DIR__, "..", "Utils", "frobenius.jl"))

# Nicolas's reference module
reference_dir = joinpath(@__DIR__, "..", "NicolasReference")

include(joinpath(reference_dir, "MySPAI.jl"))
import .MySPAI

# My Float32 implementations
spai32_dir = joinpath(@__DIR__, "..", "SPAI Iterations with Lower Precision")

include(joinpath(spai32_dir, "my_pmr_float32_spai.jl"))
include(joinpath(spai32_dir, "my_pcg_float32_spai.jl"))
include(joinpath(spai32_dir, "my_lopmr_float32_spai.jl"))

# Output directory
output_dir = "sanity_check_results_float32"
mkpath(output_dir)

matrix_name = "tri100eigs4k"

# Generate tri100eigs4k

Random.seed!(1)

n = 4_000
k = 100

A64 = spdiagm(
    0  => vcat(0.05 .+ rand(k), ones(n - k)),
    -1 => rand(n - 1)
)

A64 = sparse(A64 * A64')

# Convert the exact same matrix to Float32
A = SparseMatrixCSC{Float32, Int}(A64)

itmax = 300
tol = 0.0f0

# Jacobi preconditioner

d = diag(A)

if any(iszero, d)
    error("Cannot construct Jacobi preconditioner because A has a zero diagonal entry.")
end

Pr = spdiagm(0 => 1.0f0 ./ d)

# Initial approximate inverse M₀ = I
M0 = spdiagm(0 => ones(Float32, n))

println("Float32 sanity check")
println("Matrix:             ", matrix_name)
println("Size:               ", size(A))
println("Nonzeros:           ", nnz(A))
println("Maximum iterations: ", itmax)
println("Tolerance:          ", tol)
println("A type:             ", eltype(A))
println("Pr type:            ", eltype(Pr))
println("M0 type:            ", eltype(M0))
println()

# Run Nicolas's reference implementations

println("Running Nicolas's PMR...")

time_ref_pmr = @elapsed begin
    M_ref_pmr, R_ref_pmr =
        MySPAI.pmr(A, Pr, copy(M0), itmax, tol)
end

println(
    "Finished Nicolas's PMR in ",
    round(time_ref_pmr, digits=3),
    " seconds."
)
println()

println("Running Nicolas's PCG...")

time_ref_pcg = @elapsed begin
    M_ref_pcg, R_ref_pcg =
        MySPAI.pcg(A, Pr, copy(M0), itmax, tol)
end

println(
    "Finished Nicolas's PCG in ",
    round(time_ref_pcg, digits=3),
    " seconds."
)
println()

println("Running Nicolas's LOPMR...")

time_ref_lopmr = @elapsed begin
    M_ref_lopmr, R_ref_lopmr =
        MySPAI.lopmr(A, Pr, copy(M0), itmax, tol)
end

println(
    "Finished Nicolas's LOPMR in ",
    round(time_ref_lopmr, digits=3),
    " seconds."
)
println()

# Run my Float32 implementations

println("Running my Float32 PMR...")

time_my_pmr = @elapsed begin
    M_my_pmr, R_my_pmr =
        my_pmr_float32(A, Pr, copy(M0), itmax, tol)
end

println(
    "Finished my Float32 PMR in ",
    round(time_my_pmr, digits=3),
    " seconds."
)
println()

println("Running my Float32 PCG...")

time_my_pcg = @elapsed begin
    M_my_pcg, R_my_pcg =
        my_pcg_float32(A, Pr, copy(M0), itmax, tol)
end

println(
    "Finished my Float32 PCG in ",
    round(time_my_pcg, digits=3),
    " seconds."
)
println()

println("Running my Float32 LOPMR...")

time_my_lopmr = @elapsed begin
    M_my_lopmr, R_my_lopmr =
        my_lopmr_float32(A, Pr, copy(M0), itmax, tol)
end

println(
    "Finished my Float32 LOPMR in ",
    round(time_my_lopmr, digits=3),
    " seconds."
)
println()

# Print output types

println("Output types")

println(
    "Reference PMR:   ",
    eltype(M_ref_pmr),
    " / ",
    eltype(R_ref_pmr)
)

println(
    "Reference PCG:   ",
    eltype(M_ref_pcg),
    " / ",
    eltype(R_ref_pcg)
)

println(
    "Reference LOPMR: ",
    eltype(M_ref_lopmr),
    " / ",
    eltype(R_ref_lopmr)
)

println(
    "My PMR:          ",
    eltype(M_my_pmr),
    " / ",
    eltype(R_my_pmr)
)

println(
    "My PCG:          ",
    eltype(M_my_pcg),
    " / ",
    eltype(R_my_pcg)
)

println(
    "My LOPMR:        ",
    eltype(M_my_lopmr),
    " / ",
    eltype(R_my_lopmr)
)

println()

# Validate residual histories

function validate_residual_history(name, residuals)
    if isempty(residuals)
        error("$name returned an empty residual history.")
    end

    if any(isnan, residuals)
        @warn "$name contains NaN values."
    end

    if any(isinf, residuals)
        @warn "$name contains Inf values."
    end
end

validate_residual_history("Reference PMR", R_ref_pmr)
validate_residual_history("Reference PCG", R_ref_pcg)
validate_residual_history("Reference LOPMR", R_ref_lopmr)

validate_residual_history("My PMR", R_my_pmr)
validate_residual_history("My PCG", R_my_pcg)
validate_residual_history("My LOPMR", R_my_lopmr)

# Compare residual histories

function compare_residual_histories(name, reference, mine)
    compared_length =
        min(length(reference), length(mine))

    reference_common =
        Float64.(reference[1:compared_length])

    mine_common =
        Float64.(mine[1:compared_length])

    finite_mask =
        isfinite.(reference_common) .&
        isfinite.(mine_common)

    reference_finite =
        reference_common[finite_mask]

    mine_finite =
        mine_common[finite_mask]

    if isempty(reference_finite)
        println("$name has no common finite entries.")
        return
    end

    absolute_difference =
        abs.(reference_finite .- mine_finite)

    relative_difference =
        absolute_difference ./
        max.(
            abs.(reference_finite),
            Float64(eps(Float32))
        )


    println(name)
    println("Reference history length: ", length(reference))
    println("My history length:        ", length(mine))
    println("Compared entries:         ", compared_length)
    println()

    println(
        "Initial reference residual: ",
        reference_finite[1]
    )

    println(
        "Initial my residual:        ",
        mine_finite[1]
    )

    println()

    println(
        "Final reference residual:   ",
        reference_finite[end]
    )

    println(
        "Final my residual:          ",
        mine_finite[end]
    )

    println()

    println(
        "Minimum reference residual: ",
        minimum(reference_finite)
    )

    println(
        "Minimum my residual:        ",
        minimum(mine_finite)
    )

    println()

    println(
        "Maximum absolute difference: ",
        maximum(absolute_difference)
    )

    println(
        "Maximum relative difference: ",
        maximum(relative_difference)
    )

    println(
        "Final absolute difference:   ",
        absolute_difference[end]
    )

    println()
end

compare_residual_histories(
    "Float32 PMR comparison",
    R_ref_pmr,
    R_my_pmr
)

compare_residual_histories(
    "Float32 PCG comparison",
    R_ref_pcg,
    R_my_pcg
)

compare_residual_histories(
    "Float32 LOPMR comparison",
    R_ref_lopmr,
    R_my_lopmr
)

# Compare final approximate inverse matrices

function relative_matrix_difference(reference_matrix, my_matrix)
    reference64 = Float64.(reference_matrix)
    my64 = Float64.(my_matrix)

    denominator =
        sqrt(sum(abs2, reference64))

    numerator =
        sqrt(sum(abs2, reference64 - my64))

    return denominator == 0.0 ?
        numerator :
        numerator / denominator
end

println("Final approximate-inverse differences")

println(
    "PMR relative matrix difference: ",
    relative_matrix_difference(
        M_ref_pmr,
        M_my_pmr
    )
)

println(
    "PCG relative matrix difference: ",
    relative_matrix_difference(
        M_ref_pcg,
        M_my_pcg
    )
)

println(
    "LOPMR relative matrix difference: ",
    relative_matrix_difference(
        M_ref_lopmr,
        M_my_lopmr
    )
)

println()

# Iteration-axis helper
iteration_axis(residuals) =
    0:(length(residuals) - 1)

# Plot 1: My Float32 implementations

plot_my_methods = plot(
    iteration_axis(R_my_pmr),
    R_my_pmr,
    label = "PMR",
    color = :blue,
    linestyle = :solid,
    linewidth = 3,
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "$matrix_name — My Float32 implementations",
    legend = :outerright,
    grid = true,
    size = (1200, 700),
    dpi = 300,
    left_margin = 20 * Measures.mm,
    bottom_margin = 14 * Measures.mm,
    right_margin = 10 * Measures.mm
)

plot!(
    plot_my_methods,
    iteration_axis(R_my_pcg),
    R_my_pcg,
    label = "PCG",
    color = :red,
    linestyle = :solid,
    linewidth = 3
)

plot!(
    plot_my_methods,
    iteration_axis(R_my_lopmr),
    R_my_lopmr,
    label = "LOPMR",
    color = :green,
    linestyle = :solid,
    linewidth = 3
)

display(plot_my_methods)

savefig(
    plot_my_methods,
    joinpath(
        output_dir,
        "tri100eigs4k_float32_my_methods.png"
    )
)

# Plot 2: Reference versus mine

plot_comparison = plot(
    iteration_axis(R_ref_pmr),
    R_ref_pmr,
    label = "PMR reference",
    color = :blue,
    linestyle = :solid,
    linewidth = 3,
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "$matrix_name — Float32 sanity-check comparison",
    legend = :outerright,
    grid = true,
    size = (1200, 700),
    dpi = 300,
    left_margin = 20 * Measures.mm,
    bottom_margin = 14 * Measures.mm,
    right_margin = 10 * Measures.mm
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
    color = :orange,
    linestyle = :dash,
    linewidth = 2
)

display(plot_comparison)

savefig(
    plot_comparison,
    joinpath(
        output_dir,
        "tri100eigs4k_float32_reference_vs_mine.png"
    )
)

# Plot 3: PCG oscillation region

plot_pcg_spikes = plot(
    iteration_axis(R_ref_pcg),
    R_ref_pcg,
    label = "PCG reference",
    color = :black,
    linestyle = :solid,
    linewidth = 3,
    yscale = :log10,
    xlims = (80, 190),
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "$matrix_name — Float32 PCG oscillation region",
    legend = :topright,
    grid = true,
    size = (1200, 700),
    dpi = 300,
    left_margin = 20 * Measures.mm,
    bottom_margin = 14 * Measures.mm,
    right_margin = 10 * Measures.mm
)

plot!(
    plot_pcg_spikes,
    iteration_axis(R_my_pcg),
    R_my_pcg,
    label = "PCG mine",
    color = :red,
    linestyle = :dash,
    linewidth = 2
)

display(plot_pcg_spikes)

savefig(
    plot_pcg_spikes,
    joinpath(
        output_dir,
        "tri100eigs4k_float32_pcg_spikes.png"
    )
)

# Runtime summary

println()
println("Runtime summary")

@printf(
    "%-24s %12.3f seconds\n",
    "Reference PMR",
    time_ref_pmr
)

@printf(
    "%-24s %12.3f seconds\n",
    "My Float32 PMR",
    time_my_pmr
)

@printf(
    "%-24s %12.3f seconds\n",
    "Reference PCG",
    time_ref_pcg
)

@printf(
    "%-24s %12.3f seconds\n",
    "My Float32 PCG",
    time_my_pcg
)

@printf(
    "%-24s %12.3f seconds\n",
    "Reference LOPMR",
    time_ref_lopmr
)

@printf(
    "%-24s %12.3f seconds\n",
    "My Float32 LOPMR",
    time_my_lopmr
)

println()
println("Figures saved in: ", abspath(output_dir))
println("Float32 sanity check completed.")