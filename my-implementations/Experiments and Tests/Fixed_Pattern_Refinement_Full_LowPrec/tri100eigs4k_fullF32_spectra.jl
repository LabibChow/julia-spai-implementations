using SparseArrays
using LinearAlgebra
using Random
using Printf
using DelimitedFiles
using Plots

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")
full_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Fully Low Precision")
utils_dir = joinpath(@__DIR__, "..", "..", "Utils")

results_dir = joinpath(@__DIR__, "tri100eigs4k_fullF32_spectra_results")
plots_dir = joinpath(@__DIR__, "tri100eigs4k_fullF32_spectra_plots")
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
    S = Symmetric(0.5 .* (Matrix{Float64}(M) + transpose(Matrix{Float64}(M))))
    λ = eigvals(S)
    tol = max(1e-12 * maximum(abs, λ), eps(Float64))
    return (
        λ = λ,
        tol = tol,
        pos = count(x -> x > tol, λ),
        neg = count(x -> x < -tol, λ),
        zero = count(x -> abs(x) <= tol, λ),
    )
end

function get_spectrum(name, label, f, A, Pr, M0, itmax, tol; refine=false)
    slug = lowercase(name) * "_" * label
    path = joinpath(results_dir, "$(slug)_eigenvalues.txt")

    if isfile(path)
        λ = vec(readdlm(path, Float64))
        st = max(1e-12 * maximum(abs, λ), eps(Float64))
        return (λ=λ, tol=st,
                pos=count(x->x>st,λ),
                neg=count(x->x < -st,λ),
                zero=count(x->abs(x)<=st,λ))
    end

    println("\n$name — $label")
    M, hist = f(A, Pr, copy(M0), itmax, tol)

    if refine
        println("Refining pattern in Float64...")
        M = refine_spai_pattern(A, M)
    end

    s = spectrum(M)
    writedlm(path, s.λ)

    @printf("Iterations: %d\n", length(hist)-1)
    @printf("λmin = %.6e, λmax = %.6e\n", minimum(s.λ), maximum(s.λ))
    println("Positive: ", s.pos, "  Negative: ", s.neg, "  Near-zero: ", s.zero)

    GC.gc()
    return s
end

function rug(values, y)
    x = Float64[]
    yy = Float64[]
    for v in values
        append!(x, (abs(v), abs(v), NaN))
        append!(yy, (y-0.08, y+0.08, NaN))
    end
    return x, yy
end

function add_row!(p, s, row, sp)
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

function add_inverse!(p, λ, row, sp)
    x,y = rug(λ,row+0.14)
    plot!(p,x,y; subplot=sp,color=:black,linewidth=0.7,label=false)
end

Random.seed!(1)

n = 4000
k = 100
d = vcat(0.05 .+ rand(k), ones(n-k))
s = rand(n-1)
B = spdiagm(0 => d, -1 => s)
A = sparse(B * transpose(B))

itmax = 300
tol = 1e-15

d64 = diag(A)
d32 = Float32.(d64)

Pr64 = spdiagm(0 => 1.0 ./ d64)
Pr32 = spdiagm(0 => Float32(1) ./ d32)

M64 = spdiagm(0 => ones(Float64,n))
M32 = spdiagm(0 => ones(Float32,n))

Atri = SymTridiagonal(Vector{Float64}(diag(A)),
                      Vector{Float64}(diag(A,-1)))
λinv = 1.0 ./ eigvals(Atri)

pmr64 = get_spectrum("PMR","float64",my_pmr,A,Pr64,M64,itmax,tol)
pcg64 = get_spectrum("PCG","float64",my_pcg,A,Pr64,M64,itmax,tol)
lopmr64 = get_spectrum("LOPMR","float64",my_lopmr,A,Pr64,M64,itmax,tol)

pmr32 = get_spectrum("PMR","fullF32",my_pmr_fully_float32,A,Pr32,M32,itmax,tol)
pcg32 = get_spectrum("PCG","fullF32",my_pcg_fully_float32,A,Pr32,M32,itmax,tol)
lopmr32 = get_spectrum("LOPMR","fullF32",my_lopmr_fully_float32,A,Pr32,M32,itmax,tol)

pmr32r = get_spectrum("PMR","fullF32_refined",my_pmr_fully_float32,A,Pr32,M32,itmax,tol; refine=true)
pcg32r = get_spectrum("PCG","fullF32_refined",my_pcg_fully_float32,A,Pr32,M32,itmax,tol; refine=true)
lopmr32r = get_spectrum("LOPMR","fullF32_refined",my_lopmr_fully_float32,A,Pr32,M32,itmax,tol; refine=true)

allλ = vcat(
    abs.(λinv),
    abs.(pmr64.λ), abs.(pcg64.λ), abs.(lopmr64.λ),
    abs.(pmr32.λ), abs.(pcg32.λ), abs.(lopmr32.λ),
    abs.(pmr32r.λ), abs.(pcg32r.λ), abs.(lopmr32r.λ),
)

allλ = allλ[isfinite.(allλ) .& (allλ .> 0)]
xmin = 10.0^floor(log10(minimum(allλ)))
xmax = 10.0^ceil(log10(maximum(allλ)))

p = plot(
    layout=(3,1),
    size=(950,1050),
    dpi=300,
    plot_title="tri100eigs4k: fully reduced Float32 spectra",
)

titles = ("Float64", "Fully Float32", "Fully Float32 + Float64 refinement")
sets = (
    (pmr64,lopmr64,pcg64),
    (pmr32,lopmr32,pcg32),
    (pmr32r,lopmr32r,pcg32r),
)

for i in 1:3
    plot!(p; subplot=i, xscale=:log10, xlims=(xmin,xmax),
        ylims=(0.5,4.5),
        yticks=([4,3,2,1],["PMR","LOPMR","PCG","A⁻¹"]),
        xlabel="Eigenvalue magnitude",
        title=titles[i], legend=false, grid=true, framestyle=:box)

    add_row!(p,sets[i][1],4,i)
    add_row!(p,sets[i][2],3,i)
    add_row!(p,sets[i][3],2,i)
    add_inverse!(p,λinv,1,i)
end

plot!(p,[NaN],[NaN]; subplot=1,color=:black,linewidth=2,label="Positive eigenvalues")
plot!(p,[NaN],[NaN]; subplot=1,color=:red,linewidth=2,label="Negative eigenvalues")
plot!(p; subplot=1,legend=:outertop,legend_column=-1)

png = joinpath(plots_dir,"tri100eigs4k_fullF32_spectra.png")
pdf = joinpath(plots_dir,"tri100eigs4k_fullF32_spectra.pdf")

savefig(p,png)
savefig(p,pdf)
display(p)

println("\nSaved:")
println(png)
println(pdf)