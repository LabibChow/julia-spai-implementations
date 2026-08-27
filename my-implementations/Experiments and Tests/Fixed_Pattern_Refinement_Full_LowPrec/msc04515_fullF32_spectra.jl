using SparseArrays
using LinearAlgebra
using Serialization
using DelimitedFiles
using Printf
using Plots
using MatrixDepot

matrix_dir = joinpath(@__DIR__, "msc04515_full_lowprec_matrices")
results_dir = joinpath(@__DIR__, "msc04515_fullF32_spectra_results")
plots_dir = joinpath(@__DIR__, "msc04515_fullF32_spectra_plots")
mkpath(results_dir)
mkpath(plots_dir)

function spectrum_stats(λ)
    λ = Float64.(λ)
    t = max(1e-12 * maximum(abs,λ), eps(Float64))
    return (
        λ=λ,
        tol=t,
        pos=count(x->x>t,λ),
        neg=count(x->x < -t,λ),
        zero=count(x->abs(x)<=t,λ)
    )
end

function get_spectrum(name,file)
    path = joinpath(results_dir,"$(name)_eigenvalues.txt")

    if isfile(path)
        println("Loading cached spectrum: ",name)
        return spectrum_stats(vec(readdlm(path,Float64)))
    end

    println("\nComputing spectrum: ",name)
    M = deserialize(joinpath(matrix_dir,file))
    X = Matrix{Float64}(M)
    λ = eigvals(Symmetric(0.5 .* (X + X')))
    s = spectrum_stats(λ)

    writedlm(path,s.λ)

    @printf("λmin = %.6e, λmax = %.6e\n",minimum(s.λ),maximum(s.λ))
    println("Positive: ",s.pos,"  Negative: ",s.neg,"  Near-zero: ",s.zero)

    M = nothing
    X = nothing
    GC.gc()

    return s
end

function rug(values,y)
    x = Float64[]
    yy = Float64[]

    for v in values
        append!(x,(abs(v),abs(v),NaN))
        append!(yy,(y-0.08,y+0.08,NaN))
    end

    return x,yy
end

function add_row!(p,s,row,sp)
    pos = s.λ[s.λ .> s.tol]
    neg = s.λ[s.λ .< -s.tol]

    if !isempty(pos)
        x,y = rug(pos,row+0.14)
        plot!(p,x,y; subplot=sp,color=:black,linewidth=0.7,label=false)
    end

    if !isempty(neg)
        x,y = rug(neg,row-0.14)
        plot!(p,x,y; subplot=sp,color=:red,linewidth=0.7,label=false)
    end
end

function add_inverse!(p,λ,row,sp)
    x,y = rug(λ,row+0.14)
    plot!(p,x,y; subplot=sp,color=:black,linewidth=0.7,label=false)
end

data = mdopen("Boeing/msc04515")
A = sparse(Float64.(data.A))

inv_path = joinpath(results_dir,"A_inverse_eigenvalues.txt")

if isfile(inv_path)
    println("Loading cached A^-1 spectrum")
    λinv = vec(readdlm(inv_path,Float64))
else
    println("Computing A^-1 reference spectrum")
    λA = eigvals(Symmetric(Matrix{Float64}(A)))
    minimum(λA) <= 0 && error("A is not SPD.")
    λinv = 1.0 ./ λA
    writedlm(inv_path,λinv)
end

pmr64 = get_spectrum("pmr_float64","pmr_float64.jls")
pcg64 = get_spectrum("pcg_float64","pcg_float64.jls")
lopmr64 = get_spectrum("lopmr_float64","lopmr_float64.jls")

pmr32 = get_spectrum("pmr_fullF32","pmr_fullF32.jls")
pcg32 = get_spectrum("pcg_fullF32","pcg_fullF32.jls")
lopmr32 = get_spectrum("lopmr_fullF32","lopmr_fullF32.jls")

pmr32r = get_spectrum("pmr_fullF32_refined","pmr_fullF32_refined.jls")
pcg32r = get_spectrum("pcg_fullF32_refined","pcg_fullF32_refined.jls")
lopmr32r = get_spectrum("lopmr_fullF32_refined","lopmr_fullF32_refined.jls")

allλ = vcat(
    abs.(λinv),
    abs.(pmr64.λ),abs.(pcg64.λ),abs.(lopmr64.λ),
    abs.(pmr32.λ),abs.(pcg32.λ),abs.(lopmr32.λ),
    abs.(pmr32r.λ),abs.(pcg32r.λ),abs.(lopmr32r.λ)
)

allλ = allλ[isfinite.(allλ) .& (allλ .> 0)]
xmin = 10.0^floor(log10(minimum(allλ)))
xmax = 10.0^ceil(log10(maximum(allλ)))

p = plot(
    layout=(3,1),
    size=(950,1050),
    dpi=300,
    plot_title="msc04515: fully reduced Float32 spectra"
)

titles = (
    "Float64",
    "Fully Float32",
    "Fully Float32 + Float64 refinement"
)

sets = (
    (pmr64,lopmr64,pcg64),
    (pmr32,lopmr32,pcg32),
    (pmr32r,lopmr32r,pcg32r)
)

for i in 1:3
    plot!(p; subplot=i,
        xscale=:log10,
        xlims=(xmin,xmax),
        ylims=(0.5,4.5),
        yticks=([4,3,2,1],["PMR","LOPMR","PCG","A⁻¹"]),
        xlabel="Eigenvalue magnitude",
        title=titles[i],
        legend=false,
        grid=true,
        framestyle=:box
    )

    add_row!(p,sets[i][1],4,i)
    add_row!(p,sets[i][2],3,i)
    add_row!(p,sets[i][3],2,i)
    add_inverse!(p,λinv,1,i)
end

plot!(p,[NaN],[NaN];
    subplot=1,color=:black,linewidth=2,label="Positive eigenvalues")

plot!(p,[NaN],[NaN];
    subplot=1,color=:red,linewidth=2,label="Negative eigenvalues")

plot!(p; subplot=1,legend=:outertop,legend_column=-1)

png = joinpath(plots_dir,"msc04515_fullF32_spectra.png")
pdf = joinpath(plots_dir,"msc04515_fullF32_spectra.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

summary = joinpath(results_dir,"msc04515_fullF32_spectrum_summary.txt")

open(summary,"w") do io
    @printf(io,"%-8s %-18s %8s %8s %8s %16s %16s\n",
        "Method","Version","Pos","Neg","Zero","lambda_min","lambda_max")

    for (name,version,s) in (
        ("PMR","Float64",pmr64),
        ("PCG","Float64",pcg64),
        ("LOPMR","Float64",lopmr64),
        ("PMR","Full F32",pmr32),
        ("PCG","Full F32",pcg32),
        ("LOPMR","Full F32",lopmr32),
        ("PMR","F32 refined",pmr32r),
        ("PCG","F32 refined",pcg32r),
        ("LOPMR","F32 refined",lopmr32r)
    )
        @printf(io,"%-8s %-18s %8d %8d %8d %16.6e %16.6e\n",
            name,version,s.pos,s.neg,s.zero,
            minimum(s.λ),maximum(s.λ))
    end

    @printf(io,"\nA^-1 min: %.16e\n",minimum(λinv))
    @printf(io,"A^-1 max: %.16e\n",maximum(λinv))
end

println("\nSaved:")
println(png)
println(pdf)
println(summary)