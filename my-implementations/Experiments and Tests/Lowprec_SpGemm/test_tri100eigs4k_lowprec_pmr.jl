using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
import Measures


# Includes
include(joinpath( @__DIR__, "..", "..", "Utils", "frobenius.jl"))

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pmr_lowprec_spgemm.jl"))


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

Pr = spdiagm(
    0 => 1.0 ./ diag(A)
)

itmax = 300
tol = 1.0e-15

# Warm-up runs
println("\nCompiling full Float64 PMR...")

my_pmr(
    A,
    Pr,
    copy(M0),
    1,
    tol
)

println("\nCompiling PMR with Float32 SpGEMMs...")

my_pmr_lowprec_spgemm(
    A,
    Pr,
    copy(M0),
    1,
    tol
)

GC.gc()


# Full Float64 PMR
println("\n")
println("="^70)
println("FULL FLOAT64 PMR")
println("="^70)

full64_result = nothing

full64_total_time = @elapsed begin
    full64_result = my_pmr(
        A,
        Pr,
        copy(M0),
        itmax,
        tol
    )
end

M_full64, R_full64 = full64_result


# Float32 A*M and Float32 (Pr*A)*Z
GC.gc()

println("\n")
println("="^70)
println("PMR WITH FLOAT32 A*M AND FLOAT32 (Pr*A)*Z")
println("="^70)

lowprec_result = nothing

lowprec_total_time = @elapsed begin
    lowprec_result = my_pmr_lowprec_spgemm(
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
    "%-46s %15.6e\n",
    "Full Float64 final residual:",
    R_full64[end]
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision SpGEMM final residual:",
    R_lowprec[end]
)

@printf(
    "%-46s %15d\n",
    "Full Float64 iterations:",
    length(R_full64) - 1
)

@printf(
    "%-46s %15d\n",
    "Low-precision SpGEMM iterations:",
    length(R_lowprec) - 1
)

@printf(
    "%-46s %15.6f s\n",
    "Full Float64 total time:",
    full64_total_time
)

@printf(
    "%-46s %15.6f s\n",
    "Low-precision total time:",
    lowprec_total_time
)

@printf(
    "%-46s %15.6f\n",
    "Runtime ratio low precision / Float64:",
    lowprec_total_time / full64_total_time
)

println("\nTiming breakdown:")

@printf(
    "%-46s %15.6f s\n",
    "M -> Float32 conversion time:",
    timings.M_to_Float32
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*M SpGEMM time:",
    timings.AM_Float32_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "A*M -> Float64 conversion time:",
    timings.AM_to_Float64
)

@printf(
    "%-46s %15.6f s\n",
    "Z -> Float32 conversion time:",
    timings.Z_to_Float32
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 (Pr*A)*Z SpGEMM time:",
    timings.PrAZ_Float32_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "(Pr*A)*Z -> Float64 conversion time:",
    timings.PrAZ_to_Float64
)

@printf(
    "%-46s %15.6f s\n",
    "Total conversion time:",
    timings.total_conversion
)

@printf(
    "%-46s %15.6f s\n",
    "Total Float32 SpGEMM time:",
    timings.total_Float32_SpGEMM
)

@printf(
    "%-46s %15.6e\n",
    "nnz(M_full64)/n^2:",
    nnz(M_full64) / n^2
)

@printf(
    "%-46s %15.6e\n",
    "nnz(M_lowprec)/n^2:",
    nnz(M_lowprec) / n^2
)


# Compare convergence histories
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
    "%-46s %15.6e\n",
    "Maximum relative-history difference:",
    maximum_history_difference
)


# Plot
plot(
    0:length(relative_R_full64)-1,
    relative_R_full64,
    yscale = :log10,
    linewidth = 2,
    label = "PMR Float64",
    xlabel = "Iteration",
    ylabel = "Relative residual norm",
    title = "tri100eigs4k: Float64 vs Float32 PMR SpGEMMs",
    legend = :topright,
    grid = true,
    margin = 5Measures.mm
)

plot!(
    0:length(relative_R_lowprec)-1,
    relative_R_lowprec,
    linewidth = 2,
    linestyle = :dash,
    label = "PMR Float32 A*M and (Pr*A)*Z"
)

output_path = joinpath(
    @__DIR__,
    "tri100eigs4k_pmr_lowprec_spgemm.png"
)

savefig(output_path)

println("\nPlot saved to:")
println(output_path)