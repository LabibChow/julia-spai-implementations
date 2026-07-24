using SparseArrays
using LinearAlgebra
using Random
using Plots
using Printf
import Measures


# Includes
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_lowprec_az_ap_spai.jl"))


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
println("\nCompiling full Float64 LOPMR...")

my_lopmr(
    A,
    Pr,
    copy(M0),
    1,
    tol
)

println("\nCompiling LOPMR with Float32 A*Z and A*P...")

my_lopmr_lowprec_az_ap(
    A,
    Pr,
    copy(M0),
    1,
    tol
)

GC.gc()


# Full Float64 LOPMR
println("\n")
println("="^70)
println("FULL FLOAT64 LOPMR")
println("="^70)

full64_result = nothing

full64_total_time = @elapsed begin
    full64_result = my_lopmr(
        A,
        Pr,
        copy(M0),
        itmax,
        tol
    )
end

M_full64, R_full64 = full64_result


# Float32 A*Z and A*P, Float64 A*M
GC.gc()

println("\n")
println("="^70)
println("LOPMR: FLOAT32 A*Z AND A*P, FLOAT64 A*M")
println("="^70)

lowprec_result = nothing

lowprec_total_time = @elapsed begin
    lowprec_result = my_lopmr_lowprec_az_ap(
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
    "%-49s %15.6e\n",
    "Full Float64 final residual:",
    R_full64[end]
)

@printf(
    "%-49s %15.6e\n",
    "Float32 A*Z and A*P final residual:",
    R_lowprec[end]
)

@printf(
    "%-49s %15d\n",
    "Full Float64 iterations:",
    length(R_full64) - 1
)

@printf(
    "%-49s %15d\n",
    "Float32 A*Z and A*P iterations:",
    length(R_lowprec) - 1
)

@printf(
    "%-49s %15.6f s\n",
    "Full Float64 total time:",
    full64_total_time
)

@printf(
    "%-49s %15.6f s\n",
    "Low-precision total time:",
    lowprec_total_time
)

@printf(
    "%-49s %15.6f\n",
    "Runtime ratio low precision / Float64:",
    lowprec_total_time / full64_total_time
)

println("\nTiming breakdown:")

@printf(
    "%-49s %15.6f s\n",
    "Z -> Float32 conversion time:",
    timings.Z_to_Float32
)

@printf(
    "%-49s %15.6f s\n",
    "Float32 A*Z SpGEMM time:",
    timings.AZ_Float32_SpGEMM
)

@printf(
    "%-49s %15.6f s\n",
    "A*Z -> Float64 conversion time:",
    timings.AZ_to_Float64
)

@printf(
    "%-49s %15.6f s\n",
    "P -> Float32 conversion time:",
    timings.P_to_Float32
)

@printf(
    "%-49s %15.6f s\n",
    "Float32 A*P SpGEMM time:",
    timings.AP_Float32_SpGEMM
)

@printf(
    "%-49s %15.6f s\n",
    "A*P -> Float64 conversion time:",
    timings.AP_to_Float64
)

@printf(
    "%-49s %15.6f s\n",
    "Float64 A*M SpGEMM time:",
    timings.AM_Float64_SpGEMM
)

@printf(
    "%-49s %15.6f s\n",
    "Float64 Pr*AZ time:",
    timings.PrAZ_Float64_SpGEMM
)

@printf(
    "%-49s %15.6f s\n",
    "Float64 Pr*AP time:",
    timings.PrAP_Float64_SpGEMM
)

@printf(
    "%-49s %15.6f s\n",
    "Total conversion time:",
    timings.total_conversion
)

@printf(
    "%-49s %15.6f s\n",
    "Total Float32 SpGEMM time:",
    timings.total_Float32_SpGEMM
)

@printf(
    "%-49s %15.6f s\n",
    "Total Float64 SpGEMM time:",
    timings.total_Float64_SpGEMM
)

@printf(
    "%-49s %15.6e\n",
    "nnz(M_full64)/n^2:",
    nnz(M_full64) / n^2
)

@printf(
    "%-49s %15.6e\n",
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
    "%-49s %15.6e\n",
    "Maximum relative-history difference:",
    maximum_history_difference
)


# Plot
plot(
    0:length(relative_R_full64)-1,
    relative_R_full64,
    yscale = :log10,
    linewidth = 2,
    label = "LOPMR Float64",
    xlabel = "Iteration",
    ylabel = "Relative residual norm",
    title = "tri100eigs4k: Float32 A*Z and A*P",
    legend = :topright,
    grid = true,
    margin = 5Measures.mm
)

plot!(
    0:length(relative_R_lowprec)-1,
    relative_R_lowprec,
    linewidth = 2,
    linestyle = :dash,
    label = "LOPMR Float32 A*Z/A*P, Float64 A*M"
)

output_path = joinpath(
    @__DIR__,
    "tri100eigs4k_lopmr_lowprec_az_ap.png"
)

savefig(output_path)

println("\nPlot saved to:")
println(output_path)