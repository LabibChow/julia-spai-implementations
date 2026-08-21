using DelimitedFiles
using Plots
import Measures

# Paths
history_dir =joinpath(@__DIR__, "msc04515_nodrop_precision_histories",)
plots_dir =joinpath(@__DIR__, "msc04515_nodrop_precision_plots",)

mkpath(plots_dir,)

# Matrix information

matrix_name = "msc04515"
n = 4515

# Helper functions
function load_history(
    method,
    precision,
)

    path =
        joinpath(
            history_dir,
            "$(method)_$(precision)_history.txt",
        )


    if !isfile(path)

        error(
            "History file not found:\n$path"
        )

    end

    return vec(
        readdlm(
            path,
            Float64,
        ),
    )
end


function relative_history(
    history,
)

    return Float64.(history) ./
           sqrt(Float64(n))
end


# Protect logarithmic plots from exact zero.

function positive_values(
    values,
)

    return max.(
        Float64.(values),
        eps(Float64),
    )
end

iteration_axis(
    history,
) =
    0:(length(history) - 1)

# Load histories
pmr64 =
    relative_history(
        load_history(
            "pmr",
            "float64",
        ),
    )

pmr32 =
    relative_history(
        load_history(
            "pmr",
            "float32",
        ),
    )

pmr16 =
    relative_history(
        load_history(
            "pmr",
            "float16",
        ),
    )

pcg64 =
    relative_history(
        load_history(
            "pcg",
            "float64",
        ),
    )

pcg32 =
    relative_history(
        load_history(
            "pcg",
            "float32",
        ),
    )

pcg16 =
    relative_history(
        load_history(
            "pcg",
            "float16",
        ),
    )

lopmr64 =
    relative_history(
        load_history(
            "lopmr",
            "float64",
        ),
    )

lopmr32 =
    relative_history(
        load_history(
            "lopmr",
            "float32",
        ),
    )

lopmr16 =
    relative_history(
        load_history(
            "lopmr",
            "float16",
        ),
    )

# Data grouped by method
method_data = (

    (
        "PMR",
        pmr64,
        pmr32,
        pmr16,
    ),

    (
        "PCG",
        pcg64,
        pcg32,
        pcg16,
    ),

    (
        "LOPMR",
        lopmr64,
        lopmr32,
        lopmr16,
    ),

)

# Plot

comparison_plot =
    plot(
        layout =
            (1, 3),

        size =
            (1600, 520),

        dpi =
            300,

        plot_title =
            "msc04515 — no dropping: precision comparison",

        plot_titlefontsize =
            16,

        xlabel =
            "SPAI iteration",

        ylabel =
            "Relative Frobenius residual",

        yscale =
            :log10,

        grid =
            true,

        legend =
            false,

        left_margin =
            16 * Measures.mm,

        right_margin =
            6 * Measures.mm,

        bottom_margin =
            12 * Measures.mm,

        top_margin =
            8 * Measures.mm,
    )

# Draw PMR / PCG / LOPMR

for (
    subplot_index,
    data,
) in enumerate(
    method_data,
)

    method_name,
    history64,
    history32,
    history16 =
        data

    # Float64

    plot!(
        comparison_plot,
        iteration_axis(
            history64
        ),
        positive_values(
            history64
        );

        subplot =
            subplot_index,

        label =
            subplot_index == 1 ?
            "Float64" :
            false,

        color =
            :black,

        linewidth =
            2.2,

        linestyle =
            :solid,
    )

    # Float32

    plot!(
        comparison_plot,
        iteration_axis(
            history32
        ),
        positive_values(
            history32
        );

        subplot =
            subplot_index,

        label =
            subplot_index == 1 ?
            "Float32" :
            false,

        color =
            :blue,

        linewidth =
            2.0,

        linestyle =
            :dash,
    )

    # For msc04515 Float16 breaks essentially immediately.
    # If only iteration 0 exists, use scatter so that the
    # single point remains visible.

    if length(history16) == 1

        scatter!(
            comparison_plot,
            [0],
            [
                positive_values(
                    history16
                )[1]
            ];

            subplot =
                subplot_index,

            label =
                subplot_index == 1 ?
                "Float16" :
                false,

            color =
                :red,

            markersize =
                6,

            markerstrokewidth =
                0,
        )

    else

        plot!(
            comparison_plot,
            iteration_axis(
                history16
            ),
            positive_values(
                history16
            );

            subplot =
                subplot_index,

            label =
                subplot_index == 1 ?
                "Float16" :
                false,

            color =
                :red,

            linewidth =
                2.0,

            linestyle =
                :dot,
        )

    end


    title!(
        comparison_plot,
        method_name;

        subplot =
            subplot_index,
    )

end

# One legend only

plot!(
    comparison_plot;

    subplot =
        1,

    legend =
        :outertop,

    legend_column =
        -1,
)

# Save

plot_path =
    joinpath(
        plots_dir,
        "msc04515_nodrop_float64_float32_float16_residuals.png",
    )


savefig(
    comparison_plot,
    plot_path,
)


display(
    comparison_plot
)


println()
println("PRECISION RESIDUAL PLOT COMPLETE")

println()
println(
    "Saved to:"
)

println(
    abspath(
        plot_path
    )
)