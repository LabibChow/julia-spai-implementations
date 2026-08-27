using Plots
using DelimitedFiles
using Serialization

history_dir = joinpath(@__DIR__, "msc04515_full_lowprec_histories")
checkpoint_dir = joinpath(@__DIR__, "msc04515_nodrop_checkpoints")
plot_dir = joinpath(@__DIR__, "msc04515_full_lowprec_plots")
mkpath(plot_dir)

n = 4515
rel(x) = max.(Float64.(x) ./ sqrt(Float64(n)), eps(Float64))
load(name) = vec(readdlm(joinpath(history_dir,name)))

function refined_residual(name)
    r = deserialize(joinpath(
        checkpoint_dir,
        "$(name)_fullF32_refined_nodrop.jls"
    ))
    return r.residual
end

pmr64 = rel(load("pmr_float64_history.txt"))
pcg64 = rel(load("pcg_float64_history.txt"))
lopmr64 = rel(load("lopmr_float64_history.txt"))

pmr32 = load("pmr_fullF32_history.txt")
pcg32 = load("pcg_fullF32_history.txt")
lopmr32 = load("lopmr_fullF32_history.txt")

pmr_ref = refined_residual("pmr")
pcg_ref = refined_residual("pcg")
lopmr_ref = refined_residual("lopmr")

function panel(name,h64,h32,ref)
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

    scatter!(
        p,
        [length(h32)-1],
        [max(ref,eps(Float64))];
        label="Full F32 + Float64 refinement",
        markersize=7,
        markershape=:circle
    )

    return p
end

p = plot(
    panel("PMR",pmr64,pmr32,pmr_ref),
    panel("PCG",pcg64,pcg32,pcg_ref),
    panel("LOPMR",lopmr64,lopmr32,lopmr_ref);
    layout=(1,3),
    size=(1500,500),
    plot_title="msc04515: fully reduced Float32 and fixed-pattern refinement"
)

png = joinpath(plot_dir,"msc04515_full_refinement_comparison.png")
pdf = joinpath(plot_dir,"msc04515_full_refinement_comparison.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

println(png)
println(pdf)

println("\nRefinement marker iterations:")
println("PMR:   ",length(pmr32)-1)
println("PCG:   ",length(pcg32)-1)
println("LOPMR: ",length(lopmr32)-1)