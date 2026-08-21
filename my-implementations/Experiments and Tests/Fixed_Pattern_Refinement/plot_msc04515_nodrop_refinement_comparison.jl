using DelimitedFiles
using Plots
import Measures

# Paths
history_dir = joinpath(@__DIR__, "msc04515_nodrop_precision_histories",)
plots_dir = joinpath(@__DIR__, "msc04515_nodrop_precision_plots",)

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

function load_refined_residuals(
    method,
)

    path =
        joinpath(
            history_dir,
            "$(method)_refined_residuals.txt",
        )


    if !isfile(path)
        error(
            "Refined residual file not found:\n$path"
        )
    end

    values =
        vec(
            readdlm(
                path,
                Float64,
            ),
        )

    if length(values) < 2
        error(
            "$(method)_refined_residuals.txt " *
            "must contain two values."
        )
    end


    return (
        float32 =
            values[1],

        float16 =
            values[2],
    )
end


function relative_history(
    history,
)

    return Float64.(history) ./
           sqrt(Float64(n))
end


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

pmr64_raw =
    load_history(
        "pmr",
        "float64",
    )


pmr32_raw =
    load_history(
        "pmr",
        "float32",
    )


pmr16_raw =
    load_history(
        "pmr",
        "float16",
    )


pcg64_raw =
    load_history(
        "pcg",
        "float64",
    )


pcg32_raw =
    load_history(
        "pcg",
        "float32",
    )


pcg16_raw =
    load_history(
        "pcg",
        "float16",
    )


lopmr64_raw =
    load_history(
        "lopmr",
        "float64",
    )


lopmr32_raw =
    load_history(
        "lopmr",
        "float32",
    )


lopmr16_raw =
    load_history(
        "lopmr",
        "float16",
    )

# Convert Float64 histories to relative residuals

pmr64 =
    relative_history(
        pmr64_raw
    )


pcg64 =
    relative_history(
        pcg64_raw
    )


lopmr64 =
    relative_history(
        lopmr64_raw
    )


# Load refinement endpoints

pmr_refined =
    load_refined_residuals(
        "pmr"
    )


pcg_refined =
    load_refined_residuals(
        "pcg"
    )


lopmr_refined =
    load_refined_residuals(
        "lopmr"
    )

# Group data

# We also retain Float32 and Float16 history lengths because
# the refined marker belongs at the iteration where that
# precision's construction stopped.

method_data = (

    (
        "PMR",
        pmr64,
        pmr32_raw,
        pmr16_raw,
        pmr_refined,
    ),

    (
        "PCG",
        pcg64,
        pcg32_raw,
        pcg16_raw,
        pcg_refined,
    ),

    (
        "LOPMR",
        lopmr64,
        lopmr32_raw,
        lopmr16_raw,
        lopmr_refined,
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
            "msc04515 — no dropping: fixed-pattern refinement",

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


# Draw each method

for (
    subplot_index,
    data,
) in enumerate(
    method_data,
)

    method_name,
    history64,
    history32_raw,
    history16_raw,
    refined =
        data


    # Float64 convergence

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


    # Float32 + Float64 refinement

    float32_iteration =
        length(
            history32_raw
        ) - 1


    scatter!(
        comparison_plot,
        [
            float32_iteration
        ],
        [
            max(
                refined.float32,
                eps(Float64),
            )
        ];

        subplot =
            subplot_index,

        label =
            subplot_index == 1 ?
            "Float32 + refinement" :
            false,

        color =
            :blue,

        marker =
            :diamond,

        markersize =
            7,

        markerstrokewidth =
            0,
    )

    # For msc04515 the Float16 construction failed at the
    # beginning, so this is expected to appear at or near
    # iteration 0.

    float16_iteration =
        length(
            history16_raw
        ) - 1


    scatter!(
        comparison_plot,
        [
            float16_iteration
        ],
        [
            max(
                refined.float16,
                eps(Float64),
            )
        ];

        subplot =
            subplot_index,

        label =
            subplot_index == 1 ?
            "Float16 + refinement" :
            false,

        color =
            :red,

        marker =
            :circle,

        markersize =
            7,

        markerstrokewidth =
            0,
    )

    title!(
        comparison_plot,
        method_name;

        subplot =
            subplot_index,
    )

end


# Single horizontal legend

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
        "msc04515_nodrop_refinement_comparison.png",
    )


savefig(
    comparison_plot,
    plot_path,
)


display(
    comparison_plot
)


println()
println("REFINEMENT COMPARISON PLOT COMPLETE")

println()
println(
    "Saved to:"
)

println(
    abspath(
        plot_path
    )
)