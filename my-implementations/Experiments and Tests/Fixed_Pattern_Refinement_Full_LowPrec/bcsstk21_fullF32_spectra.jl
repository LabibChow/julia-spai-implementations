using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using Plots
using MatrixDepot

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")
full_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Fully Low Precision")
utils_dir = joinpath(@__DIR__, "..", "..", "Utils")

results_dir = joinpath(@__DIR__, "bcsstk21_fullF32_spectra_results")
plots_dir = joinpath(@__DIR__, "bcsstk21_fullF32_spectra_plots")
mkpath(results_dir)
mkpath(plots_dir)

include(joinpath(utils_dir, "frobenius.jl"))
include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))
include(joinpath(full_dir, "my_pmr_fullF32.jl"))
include(joinpath(full_dir, "my_pcg_fullF32.jl"))
include(joinpath(full_dir, "my_lopmr_fullF32.jl"))

function spectrum(M)
    X = Matrix{Float64}(M)
    λ = eigvals(Symmetric(0.5 .* (X + X')))
    t = max(1e-12 * maximum(abs,λ), eps(Float64))
    return (λ=λ, tol=t,
        pos=count(x->x>t,λ),
        neg=count(x->x < -t,λ),
        zero=count(x->abs(x)<=t,λ))
end

function load_spectrum(path)
    λ = vec(readdlm(path,Float64))
    t = max(1e-12 * maximum(abs,λ),eps(Float64))
    return (λ=λ,tol=t,
        pos=count(x->x>t,λ),
        neg=count(x->x < -t,λ),
        zero=count(x->abs(x)<=t,λ))
end

function get64(name,f,A,Pr,M0,itmax,tol)
    path = joinpath(results_dir,"$(lowercase(name))_float64_eigenvalues.txt")
    isfile(path) && return load_spectrum(path)

    println("\n$name Float64")
    M,h = f(A,Pr,copy(M0),itmax,tol)
    s = spectrum(M)
    writedlm(path,s.λ)

    @printf("Iterations: %d\n",length(h)-1)
    @printf("λmin = %.6e, λmax = %.6e\n",minimum(s.λ),maximum(s.λ))
    println("Positive: ",s.pos,"  Negative: ",s.neg,"  Near-zero: ",s.zero)
    GC.gc()
    return s
end

function get32_pair(name,f,A,Pr,M0,itmax,tol)
    slug = lowercase(name)
    p1 = joinpath(results_dir,"$(slug)_fullF32_eigenvalues.txt")
    p2 = joinpath(results_dir,"$(slug)_fullF32_refined_eigenvalues.txt")

    if isfile(p1) && isfile(p2)
        return load_spectrum(p1), load_spectrum(p2)
    end

    println("\n$name Fully Float32")
    M,h = f(A,Pr,copy(M0),itmax,tol)
    s = spectrum(M)
    writedlm(p1,s.λ)

    @printf("Iterations: %d\n",length(h)-1)
    @printf("F32 λmin = %.6e, λmax = %.6e\n",minimum(s.λ),maximum(s.λ))

    println("Refining $name pattern...")
    Mref = refine_spai_pattern(A,M)
    sr = spectrum(Mref)
    writedlm(p2,sr.λ)

    @printf("Refined λmin = %.6e, λmax = %.6e\n",
        minimum(sr.λ),maximum(sr.λ))
    GC.gc()

    return s,sr
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

data = mdopen("HB/bcsstk21")
A = sparse(Float64.(data.A))
n = size(A,1)

itmax = 1000
tol = 1e-15

d64 = diag(A)
d32 = Float32.(d64)

Pr64 = spdiagm(0 => 1.0 ./ d64)
Pr32 = spdiagm(0 => Float32(1) ./ d32)

M64 = spdiagm(0 => ones(Float64,n))
M32 = spdiagm(0 => ones(Float32,n))

println("Matrix: bcsstk21")
println("Size: ",size(A))
println("nnz(A): ",nnz(A))
println("dropping: NONE")

inv_path = joinpath(results_dir,"A_inverse_eigenvalues.txt")

if isfile(inv_path)
    λinv = vec(readdlm(inv_path,Float64))
else
    println("\nComputing A^-1 reference spectrum...")
    λA = eigvals(Symmetric(Matrix{Float64}(A)))
    minimum(λA) <= 0 && error("A is not SPD.")
    λinv = 1.0 ./ λA
    writedlm(inv_path,λinv)
end

pmr64 = get64("PMR",my_pmr,A,Pr64,M64,itmax,tol)
GC.gc()
pcg64 = get64("PCG",my_pcg,A,Pr64,M64,itmax,tol)
GC.gc()
lopmr64 = get64("LOPMR",my_lopmr,A,Pr64,M64,itmax,tol)

GC.gc()
pmr32,pmr32r = get32_pair("PMR",my_pmr_fully_float32,A,Pr32,M32,itmax,tol)
GC.gc()
pcg32,pcg32r = get32_pair("PCG",my_pcg_fully_float32,A,Pr32,M32,itmax,tol)
GC.gc()
lopmr32,lopmr32r = get32_pair("LOPMR",my_lopmr_fully_float32,A,Pr32,M32,itmax,tol)

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
    plot_title="bcsstk21: fully reduced Float32 spectra"
)

titles = ("Float64","Fully Float32","Fully Float32 + Float64 refinement")
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
        framestyle=:box)

    add_row!(p,sets[i][1],4,i)
    add_row!(p,sets[i][2],3,i)
    add_row!(p,sets[i][3],2,i)
    add_inverse!(p,λinv,1,i)
end

plot!(p,[NaN],[NaN]; subplot=1,color=:black,linewidth=2,label="Positive eigenvalues")
plot!(p,[NaN],[NaN]; subplot=1,color=:red,linewidth=2,label="Negative eigenvalues")
plot!(p; subplot=1,legend=:outertop,legend_column=-1)

png = joinpath(plots_dir,"bcsstk21_fullF32_spectra.png")
pdf = joinpath(plots_dir,"bcsstk21_fullF32_spectra.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

summary = joinpath(results_dir,"bcsstk21_fullF32_spectrum_summary.txt")

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
            name,version,s.pos,s.neg,s.zero,minimum(s.λ),maximum(s.λ))
    end

    @printf(io,"\nA^-1 min: %.16e\n",minimum(λinv))
    @printf(io,"A^-1 max: %.16e\n",maximum(λinv))
end

println("\nSaved:")
println(png)
println(pdf)
println(summary)