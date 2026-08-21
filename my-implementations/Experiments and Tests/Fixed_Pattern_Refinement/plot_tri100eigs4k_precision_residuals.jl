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


# Load histories
pmr64 = vec(readdlm(
    joinpath(history_dir, "pmr_float64_history.txt")
))

pmr32 = vec(readdlm(
    joinpath(history_dir, "pmr_float32_history.txt")
))

pmr16 = vec(readdlm(
    joinpath(history_dir, "pmr_float16_history.txt")
))


pcg64 = vec(readdlm(
    joinpath(history_dir, "pcg_float64_history.txt")
))

pcg32 = vec(readdlm(
    joinpath(history_dir, "pcg_float32_history.txt")
))

pcg16 = vec(readdlm(
    joinpath(history_dir, "pcg_float16_history.txt")
))


lopmr64 = vec(readdlm(
    joinpath(history_dir, "lopmr_float64_history.txt")
))

lopmr32 = vec(readdlm(
    joinpath(history_dir, "lopmr_float32_history.txt")
))

lopmr16 = vec(readdlm(
    joinpath(history_dir, "lopmr_float16_history.txt")
))


# Convert absolute residuals to relative residuals

pmr64 = positive_values(
    relative_residual_history(pmr64, n)
)

pmr32 = positive_values(
    relative_residual_history(pmr32, n)
)

pmr16 = positive_values(
    relative_residual_history(pmr16, n)
)


pcg64 = positive_values(
    relative_residual_history(pcg64, n)
)

pcg32 = positive_values(
    relative_residual_history(pcg32, n)
)

pcg16 = positive_values(
    relative_residual_history(pcg16, n)
)


lopmr64 = positive_values(
    relative_residual_history(lopmr64, n)
)

lopmr32 = positive_values(
    relative_residual_history(lopmr32, n)
)

lopmr16 = positive_values(
    relative_residual_history(lopmr16, n)
)


# ============================================================
# Plot helper
# ============================================================

function add_precision_plot!(
    plot_object,
    subplot_index,
    method_name,
    history64,
    history32,
    history16,
)

    iterations64 = 0:(length(history64) - 1)
    iterations32 = 0:(length(history32) - 1)
    iterations16 = 0:(length(history16) - 1)

    # Float64
    plot!(
        plot_object,
        iterations64,
        history64;
        subplot = subplot_index,
        label = "Float64",
        linewidth = 2,
    )

    # Float32
    plot!(
        plot_object,
        iterations32,
        history32;
        subplot = subplot_index,
        label = "Float32",
        linewidth = 2,
        linestyle = :dash,
    )

    # Float16
    plot!(
        plot_object,
        iterations16,
        history16;
        subplot = subplot_index,
        label = "Float16",
        linewidth = 2,
        linestyle = :dot,
    )

    title!(
        plot_object,
        method_name;
        subplot = subplot_index,
    )

    return nothing
end

# Build figure
comparison_plot = plot(
    layout = (1, 3),

    size = (1500, 500),

    xlabel = "SPAI iteration",

    ylabel = "Relative Frobenius residual",

    yscale = :log10,

    legend = :topright,

    grid = true,

    plot_title =
        "tri100eigs4k: Float64, Float32, and Float16 convergence",
)


# PMR
add_precision_plot!(
    comparison_plot,
    1,
    "PMR",
    pmr64,
    pmr32,
    pmr16,
)


# PCG
add_precision_plot!(
    comparison_plot,
    2,
    "PCG",
    pcg64,
    pcg32,
    pcg16,
)


# LOPMR
add_precision_plot!(
    comparison_plot,
    3,
    "LOPMR",
    lopmr64,
    lopmr32,
    lopmr16,
)


# Save

png_path = joinpath(
    plot_dir,
    "tri100eigs4k_float64_float32_float16_convergence.png",
)

pdf_path = joinpath(
    plot_dir,
    "tri100eigs4k_float64_float32_float16_convergence.pdf",
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

println()
println("Plots saved to:")
println(png_path)
println(pdf_path)