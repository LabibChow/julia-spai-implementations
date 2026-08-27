using Plots
using DelimitedFiles

history_dir = joinpath(@__DIR__, "tri100eigs4k_full_lowprec_histories")
plot_dir = joinpath(@__DIR__, "tri100eigs4k_full_lowprec_plots")
mkpath(plot_dir)

n = 4000

relative_history(x) = max.(Float64.(x) ./ sqrt(Float64(n)), eps(Float64))

function load_history(name)
    return vec(readdlm(joinpath(history_dir, name)))
end

function load_refined(name)
    values = vec(readdlm(joinpath(history_dir, name)))
    return Float64(values[1]), Float64(values[2])
end

pmr64 = relative_history(load_history("pmr_float64_history.txt"))
pmr32 = load_history("pmr_fullF32_history.txt")
pmr16 = load_history("pmr_fullF16_history.txt")
pmr32_ref, pmr16_ref = load_refined("pmr_full_refined_residuals.txt")

pcg64 = relative_history(load_history("pcg_float64_history.txt"))
pcg32 = load_history("pcg_fullF32_history.txt")
pcg16 = load_history("pcg_fullF16_history.txt")
pcg32_ref, pcg16_ref = load_refined("pcg_full_refined_residuals.txt")

lopmr64 = relative_history(load_history("lopmr_float64_history.txt"))
lopmr32 = load_history("lopmr_fullF32_history.txt")
lopmr16 = load_history("lopmr_fullF16_history.txt")
lopmr32_ref, lopmr16_ref = load_refined("lopmr_full_refined_residuals.txt")

function add_plot!(p, i, name, h64, h32, h16, ref32, ref16)
    plot!(p, 0:length(h64)-1, h64;
        subplot=i, label="Float64", linewidth=2)

    scatter!(p, [length(h32)-1], [max(ref32, eps(Float64))];
        subplot=i,
        label="Full F32 + Float64 refinement",
        markersize=7,
        markershape=:circle)

    scatter!(p, [length(h16)-1], [max(ref16, eps(Float64))];
        subplot=i,
        label="Full F16 + Float64 refinement",
        markersize=7,
        markershape=:diamond)

    title!(p, name; subplot=i)
end

p = plot(
    layout=(1,3),
    size=(1500,500),
    xlabel="SPAI iteration",
    ylabel="Relative Frobenius residual",
    yscale=:log10,
    legend=:topright,
    grid=true,
    plot_title="tri100eigs4k: fully reduced precision and fixed-pattern refinement",
)

add_plot!(p, 1, "PMR",
    pmr64, pmr32, pmr16, pmr32_ref, pmr16_ref)

add_plot!(p, 2, "PCG",
    pcg64, pcg32, pcg16, pcg32_ref, pcg16_ref)

add_plot!(p, 3, "LOPMR",
    lopmr64, lopmr32, lopmr16, lopmr32_ref, lopmr16_ref)

png_path = joinpath(plot_dir, "tri100eigs4k_full_refinement_comparison.png")
pdf_path = joinpath(plot_dir, "tri100eigs4k_full_refinement_comparison.pdf")

savefig(p, png_path)
savefig(p, pdf_path)
display(p)

println("Saved to:")
println(png_path)
println(pdf_path)

println("Refinement marker iterations:")
println("PMR:   F32 = ", length(pmr32)-1, ", F16 = ", length(pmr16)-1)
println("PCG:   F32 = ", length(pcg32)-1, ", F16 = ", length(pcg16)-1)
println("LOPMR: F32 = ", length(lopmr32)-1, ", F16 = ", length(lopmr16)-1)