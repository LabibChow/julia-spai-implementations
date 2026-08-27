using Plots
using DelimitedFiles

history_dir = joinpath(@__DIR__, "tri100eigs4k_full_lowprec_histories")
plot_dir = joinpath(@__DIR__, "tri100eigs4k_full_lowprec_plots")
mkpath(plot_dir)

n = 4000

relative_history(x) = Float64.(x) ./ sqrt(Float64(n))

function load_history(name)
    vec(readdlm(joinpath(history_dir, name)))
end


pmr64 = relative_history(load_history("pmr_float64_history.txt"))
pmr32 = relative_history(load_history("pmr_fullF32_history.txt"))
pmr16 = relative_history(load_history("pmr_fullF16_history.txt"))

pcg64 = max.(relative_history(load_history("pcg_float64_history.txt")), eps(Float64))
pcg32 = max.(relative_history(load_history("pcg_fullF32_history.txt")), eps(Float64))
pcg16 = max.(relative_history(load_history("pcg_fullF16_history.txt")), eps(Float64))

lopmr64 = max.(relative_history(load_history("lopmr_float64_history.txt")), eps(Float64))
lopmr32 = max.(relative_history(load_history("lopmr_fullF32_history.txt")), eps(Float64))
lopmr16 = max.(relative_history(load_history("lopmr_fullF16_history.txt")), eps(Float64))

p1 = plot(
    0:length(pmr64)-1, pmr64,
    label="Float64", linewidth=2,
    xlabel="SPAI iteration",
    ylabel="Relative Frobenius residual",
    title="PMR",
    grid=true,
)

plot!(p1, 0:length(pmr32)-1, pmr32;
    label="Fully Float32", linewidth=2, linestyle=:dash)

plot!(p1, 0:length(pmr16)-1, pmr16;
    label="Fully Float16", linewidth=2, linestyle=:dot)

p2 = plot(
    0:length(pcg64)-1, pcg64,
    label="Float64", linewidth=2,
    xlabel="SPAI iteration",
    ylabel="Relative Frobenius residual",
    title="PCG",
    yscale=:log10,
    grid=true,
)

plot!(p2, 0:length(pcg32)-1, pcg32;
    label="Fully Float32", linewidth=2, linestyle=:dash)

plot!(p2, 0:length(pcg16)-1, pcg16;
    label="Fully Float16", linewidth=2, linestyle=:dot)

p3 = plot(
    0:length(lopmr64)-1, lopmr64,
    label="Float64", linewidth=2,
    xlabel="SPAI iteration",
    ylabel="Relative Frobenius residual",
    title="LOPMR",
    yscale=:log10,
    grid=true,
)

plot!(p3, 0:length(lopmr32)-1, lopmr32;
    label="Fully Float32", linewidth=2, linestyle=:dash)

plot!(p3, 0:length(lopmr16)-1, lopmr16;
    label="Fully Float16", linewidth=2, linestyle=:dot)

p = plot(
    p1, p2, p3,
    layout=(1,3),
    size=(1500,500),
    plot_title="tri100eigs4k: fully reduced precision convergence",
)

png_path = joinpath(plot_dir, "tri100eigs4k_fully_reduced_convergence.png")
pdf_path = joinpath(plot_dir, "tri100eigs4k_fully_reduced_convergence.pdf")

savefig(p, png_path)
savefig(p, pdf_path)
display(p)

println("Saved to:")
println(png_path)
println(pdf_path)