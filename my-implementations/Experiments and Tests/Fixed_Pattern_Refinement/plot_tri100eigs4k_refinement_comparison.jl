using Plots
using DelimitedFiles

# Paths
history_dir = joinpath(@__DIR__, "tri100eigs4k_precision_histories",)
plot_dir = joinpath(@__DIR__, "tri100eigs4k_precision_plots",)

mkpath(plot_dir)

# Matrix size
n = 4000

# Helper functions
function relative_residual_history(history, n)
    return Float64.(history) ./ sqrt(Float64(n))
end

function positive_values(values)
    return max.(
        Float64.(values),
        eps(Float64),
    )
end


# Load Float64 histories
pmr64 = vec(readdlm(
    joinpath(
        history_dir,
        "pmr_float64_history.txt",
    ),
))

pcg64 = vec(readdlm(
    joinpath(
        history_dir,
        "pcg_float64_history.txt",
    ),
))

lopmr64 = vec(readdlm(
    joinpath(
        history_dir,
        "lopmr_float64_history.txt",
    ),
))


# Load Float32 histories
pmr32 = vec(readdlm(
    joinpath(
        history_dir,
        "pmr_float32_history.txt",
    ),
))

pcg32 = vec(readdlm(
    joinpath(
        history_dir,
        "pcg_float32_history.txt",
    ),
))

lopmr32 = vec(readdlm(
    joinpath(
        history_dir,
        "lopmr_float32_history.txt",
    ),
))

# This is particularly important for PCG because Float16 PCG
# terminated early after numerical breakdown.
pmr16 = vec(readdlm(
    joinpath(
        history_dir,
        "pmr_float16_history.txt",
    ),
))

pcg16 = vec(readdlm(
    joinpath(
        history_dir,
        "pcg_float16_history.txt",
    ),
))

lopmr16 = vec(readdlm(
    joinpath(
        history_dir,
        "lopmr_float16_history.txt",
    ),
))


# Convert Float64 histories to relative Frobenius residuals
pmr64_rel = positive_values(
    relative_residual_history(
        pmr64,
        n,
    ),
)

pcg64_rel = positive_values(
    relative_residual_history(
        pcg64,
        n,
    ),
)

lopmr64_rel = positive_values(
    relative_residual_history(
        lopmr64,
        n,
    ),
)

# Refined residual values
# Float32 pattern -> Float64 fixed-pattern refinement

pmr32_refined = 3.241058e-02
pcg32_refined = 9.211613e-11
lopmr32_refined = 9.113905e-11


# Float16 pattern -> Float64 fixed-pattern refinement

pmr16_refined = 1.070870e-01
pcg16_refined = 2.938304e-04
lopmr16_refined = 7.326806e-03


# Plot helper

function add_refinement_plot!(
    plot_object,
    subplot_index,
    method_name,
    history64,
    history32,
    history16,
    refined32,
    refined16,
)

    # Iteration numbers

    iterations64 = 0:(length(history64) - 1)

    final_iteration32 =
        length(history32) - 1

    final_iteration16 =
        length(history16) - 1


    # Fully Float64 convergence curve

    plot!(
        plot_object,
        iterations64,
        history64;
        subplot = subplot_index,
        label = "Float64",
        linewidth = 2,
    )


    # Float32 pattern + Float64 refinement

    scatter!(
        plot_object,
        [final_iteration32],
        [max(refined32, eps(Float64))];
        subplot = subplot_index,
        label = "Float32 pattern + Float64 refinement",
        markersize = 7,
        markershape = :circle,
    )


    # Float16 pattern + Float64 refinement

    scatter!(
        plot_object,
        [final_iteration16],
        [max(refined16, eps(Float64))];
        subplot = subplot_index,
        label = "Float16 pattern + Float64 refinement",
        markersize = 7,
        markershape = :diamond,
    )

    # Subplot title

    title!(
        plot_object,
        method_name;
        subplot = subplot_index,
    )

    return nothing
end


# Main figure
comparison_plot = plot(
    layout = (1, 3),

    size = (1500, 500),

    xlabel = "SPAI iteration",

    ylabel = "Relative Frobenius residual",

    yscale = :log10,

    legend = :topright,

    grid = true,

    plot_title =
        "tri100eigs4k: Float64 and fixed-pattern refinement",
)

# PMR
add_refinement_plot!(
    comparison_plot,
    1,
    "PMR",
    pmr64_rel,
    pmr32,
    pmr16,
    pmr32_refined,
    pmr16_refined,
)

# PCG
add_refinement_plot!(
    comparison_plot,
    2,
    "PCG",
    pcg64_rel,
    pcg32,
    pcg16,
    pcg32_refined,
    pcg16_refined,
)

# LOPMR
add_refinement_plot!(
    comparison_plot,
    3,
    "LOPMR",
    lopmr64_rel,
    lopmr32,
    lopmr16,
    lopmr32_refined,
    lopmr16_refined,
)


# Save figure
png_path = joinpath(
    plot_dir,
    "tri100eigs4k_refinement_comparison.png",
)

pdf_path = joinpath(
    plot_dir,
    "tri100eigs4k_refinement_comparison.pdf",
)

savefig(
    comparison_plot,
    png_path,
)

savefig(
    comparison_plot,
    pdf_path,
)

display(comparison_plot)


# Output
println()
println("Refinement comparison plots saved to:")
println(png_path)
println(pdf_path)

println()
println("Refinement marker iterations:")

println(
    "PMR:   Float32 = ",
    length(pmr32) - 1,
    ", Float16 = ",
    length(pmr16) - 1,
)

println(
    "PCG:   Float32 = ",
    length(pcg32) - 1,
    ", Float16 = ",
    length(pcg16) - 1,
)

println(
    "LOPMR: Float32 = ",
    length(lopmr32) - 1,
    ", Float16 = ",
    length(lopmr16) - 1,
)