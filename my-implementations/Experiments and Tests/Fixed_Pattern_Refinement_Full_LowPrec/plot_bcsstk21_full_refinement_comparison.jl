using Plots
using DelimitedFiles

history_dir = joinpath(@__DIR__, "bcsstk21_full_lowprec_histories")
plot_dir = joinpath(@__DIR__, "bcsstk21_full_lowprec_plots")
mkpath(plot_dir)

n = 3600
rel(x) = max.(Float64.(x) ./ sqrt(Float64(n)),eps(Float64))
load(name) = vec(readdlm(joinpath(history_dir,name)))

pmr64 = rel(load("pmr_float64_history.txt"))
pcg64 = rel(load("pcg_float64_history.txt"))
lopmr64 = rel(load("lopmr_float64_history.txt"))

pmr32 = load("pmr_fullF32_history.txt")
pmr16 = load("pmr_fullF16_history.txt")
pcg32 = load("pcg_fullF32_history.txt")
pcg16 = load("pcg_fullF16_history.txt")
lopmr32 = load("lopmr_fullF32_history.txt")
lopmr16 = load("lopmr_fullF16_history.txt")

pmr_ref = load("pmr_full_refined_residuals.txt")
pcg_ref = load("pcg_full_refined_residuals.txt")
lopmr_ref = load("lopmr_full_refined_residuals.txt")

function panel(name,h64,h32,h16,ref)
    p = plot(0:length(h64)-1,h64;
        label="Float64",linewidth=2,
        xlabel="SPAI iteration",
        ylabel="Relative Frobenius residual",
        title=name,yscale=:log10,grid=true)

    scatter!(p,[length(h32)-1],[ref[1]];
        label="Full F32 + Float64 refinement",
        markersize=7,markershape=:circle)

    scatter!(p,[length(h16)-1],[ref[2]];
        label="Full F16 + Float64 refinement",
        markersize=7,markershape=:diamond)

    return p
end

p = plot(
    panel("PMR",pmr64,pmr32,pmr16,pmr_ref),
    panel("PCG",pcg64,pcg32,pcg16,pcg_ref),
    panel("LOPMR",lopmr64,lopmr32,lopmr16,lopmr_ref);
    layout=(1,3),
    size=(1500,500),
    plot_title="bcsstk21: fully reduced precision and fixed-pattern refinement"
)

png = joinpath(plot_dir,"bcsstk21_full_refinement_comparison.png")
pdf = joinpath(plot_dir,"bcsstk21_full_refinement_comparison.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

println(png)
println(pdf)