using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
import Measures


include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_pcg_lowprec_ap_spai.jl"))


# Construct tri100eigs4k
Random.seed!(1)

n = 4000
k = 100

main_diagonal = [
    0.05 .+ rand(k)
    ones(n - k)
]

lower_diagonal = rand(n - 1)

B = spdiagm(
     0 => main_diagonal,
    -1 => lower_diagonal
)

A = B * transpose(B)
A = SparseMatrixCSC{Float64, Int}(A)

println("Matrix: tri100eigs4k")
println("size(A) = $(size(A))")
println("nnz(A) = $(nnz(A))")
println("eltype(A) = $(eltype(A))")


# Initial approximate inverse and Jacobi preconditioner
M0 = spdiagm(0 => ones(Float64, n))
Pr = spdiagm(0 => 1.0 ./ diag(A))

itmax = 300
tol = 1.0e-15


# Warm-up
println("\nCompiling full Float64 PCG...")
my_pcg(A, Pr, copy(M0), 1, tol)

println("\nCompiling PCG with Float32 A*P...")
my_pcg_lowprec_ap(A, Pr, copy(M0), 1, tol)

GC.gc()

# Full Float64 PCG
println("\n")
println("="^70)
println("FULL FLOAT64 PCG")
println("="^70)

full64_result = nothing

full64_total_time = @elapsed begin
    full64_result = my_pcg(
        A,
        Pr,
        copy(M0),
        itmax,
        tol
    )
end

M_full64, R_full64 = full64_result


# Float32 A*P, Float64 A*M
GC.gc()

println("\n")
println("="^70)
println("PCG WITH FLOAT32 A*P AND FLOAT64 A*M")
println("="^70)

lowprec_result = nothing

lowprec_total_time = @elapsed begin
    lowprec_result = my_pcg_lowprec_ap(
        A,
        Pr,
        copy(M0),
        itmax,
        tol
    )
end

M_lowprec, R_lowprec, timings = lowprec_result


# Relative residual histories
relative_R_full64 = R_full64 ./ R_full64[1]
relative_R_lowprec = R_lowprec ./ R_lowprec[1]


# Summary
println("\n")
println("="^70)
println("EXPERIMENT SUMMARY")
println("="^70)

@printf(
    "%-42s %15.6e\n",
    "Full Float64 final residual:",
    R_full64[end]
)

@printf(
    "%-42s %15.6e\n",
    "Float32 A*P final residual:",
    R_lowprec[end]
)

@printf(
    "%-42s %15d\n",
    "Full Float64 iterations:",
    length(R_full64) - 1
)

@printf(
    "%-42s %15d\n",
    "Float32 A*P iterations:",
    length(R_lowprec) - 1
)

@printf(
    "%-42s %15.6f s\n",
    "Full Float64 total time:",
    full64_total_time
)

@printf(
    "%-42s %15.6f s\n",
    "Float32 A*P total time:",
    lowprec_total_time
)

@printf(
    "%-42s %15.6f\n",
    "Runtime ratio low precision / Float64:",
    lowprec_total_time / full64_total_time
)

@printf(
    "%-42s %15.6f s\n",
    "P -> Float32 conversion time:",
    timings.P_to_Float32
)

@printf(
    "%-42s %15.6f s\n",
    "Float32 A*P SpGEMM time:",
    timings.AP_Float32_SpGEMM
)

@printf(
    "%-42s %15.6f s\n",
    "A*P -> Float64 conversion time:",
    timings.AP_to_Float64
)

@printf(
    "%-42s %15.6f s\n",
    "Total A*P operation time:",
    timings.total_AP_operations
)

@printf(
    "%-42s %15.6f s\n",
    "Float64 A*M SpGEMM time:",
    timings.AM_Float64_SpGEMM
)

@printf(
    "%-42s %15.6e\n",
    "nnz(M_full64)/n^2:",
    nnz(M_full64) / n^2
)

@printf(
    "%-42s %15.6e\n",
    "nnz(M_lowprec)/n^2:",
    nnz(M_lowprec) / n^2
)


# Compare histories
common_length = min(
    length(relative_R_full64),
    length(relative_R_lowprec)
)

maximum_history_difference = maximum(
    abs.(
        relative_R_full64[1:common_length] .-
        relative_R_lowprec[1:common_length]
    )
)

@printf(
    "%-42s %15.6e\n",
    "Maximum relative-history difference:",
    maximum_history_difference
)


# Plot
plot(
    0:length(relative_R_full64)-1,
    relative_R_full64,
    yscale = :log10,
    linewidth = 2,
    label = "PCG Float64",
    xlabel = "Iteration",
    ylabel = "Relative residual norm",
    title = "tri100eigs4k: Float64 vs Float32 A*P",
    legend = :topright,
    grid = true,
    margin = 5Measures.mm
)

plot!(
    0:length(relative_R_lowprec)-1,
    relative_R_lowprec,
    linewidth = 2,
    linestyle = :dash,
    label = "PCG Float32 A*P, Float64 A*M"
)

output_path = joinpath(
    @__DIR__,
    "tri100eigs4k_pcg_lowprec_ap.png"
)

savefig(output_path)

println("\nPlot saved to:")
println(output_path)