using Plots
using DelimitedFiles

history_dir = joinpath(@__DIR__, "bcsstk21_full_lowprec_histories")
plot_dir = joinpath(@__DIR__, "bcsstk21_full_lowprec_plots")
mkpath(plot_dir)

n = 3600
rel(x) = max.(Float64.(x) ./ sqrt(Float64(n)),eps(Float64))
load(dir,name) = vec(readdlm(joinpath(dir,name)))

pmr64 = rel(load(history_dir,"pmr_float64_history.txt"))
pmr32 = rel(load(history_dir,"pmr_fullF32_history.txt"))
pmr16 = rel(load(history_dir,"pmr_fullF16_history.txt"))

pcg64 = rel(load(history_dir,"pcg_float64_history.txt"))
pcg32 = rel(load(history_dir,"pcg_fullF32_history.txt"))
pcg16 = rel(load(history_dir,"pcg_fullF16_history.txt"))

lopmr64 = rel(load(history_dir,"lopmr_float64_history.txt"))
lopmr32 = rel(load(history_dir,"lopmr_fullF32_history.txt"))
lopmr16 = rel(load(history_dir,"lopmr_fullF16_history.txt"))

function panel(name,h64,h32,h16)
    p = plot(0:length(h64)-1,h64;
        label="Float64",linewidth=2,
        xlabel="SPAI iteration",
        ylabel="Relative Frobenius residual",
        title=name,yscale=:log10,grid=true)

    plot!(p,0:length(h32)-1,h32;
        label="Fully Float32",linewidth=2,linestyle=:dash)

    plot!(p,0:length(h16)-1,h16;
        label="Fully Float16",linewidth=2,linestyle=:dot)

    return p
end

p = plot(
    panel("PMR",pmr64,pmr32,pmr16),
    panel("PCG",pcg64,pcg32,pcg16),
    panel("LOPMR",lopmr64,lopmr32,lopmr16);
    layout=(1,3),
    size=(1500,500),
    plot_title="bcsstk21: fully reduced precision convergence"
)

png = joinpath(plot_dir,"bcsstk21_fully_reduced_convergence.png")
pdf = joinpath(plot_dir,"bcsstk21_fully_reduced_convergence.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

println(png)
println(pdf)