using Plots
using DelimitedFiles

history_dir = joinpath(@__DIR__, "msc04515_full_lowprec_histories")
plot_dir = joinpath(@__DIR__, "msc04515_full_lowprec_plots")
mkpath(plot_dir)

n = 4515
rel(x) = max.(Float64.(x) ./ sqrt(Float64(n)), eps(Float64))
load(name) = vec(readdlm(joinpath(history_dir,name)))

pmr64 = rel(load("pmr_float64_history.txt"))
pmr32 = rel(load("pmr_fullF32_history.txt"))

pcg64 = rel(load("pcg_float64_history.txt"))
pcg32 = rel(load("pcg_fullF32_history.txt"))

lopmr64 = rel(load("lopmr_float64_history.txt"))
lopmr32 = rel(load("lopmr_fullF32_history.txt"))

function panel(name,h64,h32)
    p = plot(
        0:length(h64)-1,h64;
        label="Float64",
        linewidth=2,
        xlabel="SPAI iteration",
        ylabel="Relative Frobenius residual",
        title=name,
        yscale=:log10,
        grid=true
    )

    plot!(
        p,0:length(h32)-1,h32;
        label="Fully Float32",
        linewidth=2,
        linestyle=:dash
    )

    return p
end

p = plot(
    panel("PMR",pmr64,pmr32),
    panel("PCG",pcg64,pcg32),
    panel("LOPMR",lopmr64,lopmr32);
    layout=(1,3),
    size=(1500,500),
    plot_title="msc04515: fully reduced Float32 convergence"
)

png = joinpath(plot_dir,"msc04515_fully_reduced_convergence.png")
pdf = joinpath(plot_dir,"msc04515_fully_reduced_convergence.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

println(png)
println(pdf)