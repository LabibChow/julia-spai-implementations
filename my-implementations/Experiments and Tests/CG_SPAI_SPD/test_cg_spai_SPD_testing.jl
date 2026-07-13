using MatrixDepot
using SparseArrays
using LinearAlgebra
using Plots

include(joinpath(@__DIR__, "..", "..", "Utils", "check_spd.jl"))
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))
include(joinpath(@__DIR__, "..", "..", "SPAI Iterations", "my_cg_spai.jl"))


# 1. Load HB/bcsstk21

md = mdopen("HB/bcsstk21")

A64 = sparse(Float64.(md.A))
A32 = sparse(Float32.(md.A))

n = size(A64, 1)

println("FLOAT64 VS FLOAT32 CG-SPAI")
println("Matrix: HB/bcsstk21")
println("Size: ", size(A64))
println("Nonzeros in A: ", nnz(A64))

# Check whether the original matrix A is SPD

println("\nChecking whether original matrix A64 is SPD...")

spd_A64 = check_spd(A64; name = "HB/bcsstk21")

if !spd_A64.is_spd
    error("HB/bcsstk21 is not SPD, so the SPAI experiment will not be run.")
end

println("HB/bcsstk21 passed the SPD check.")

# 2. Common SPAI settings

spai_itmax = 75
spai_tol = 1e-3

# 3. Build Float64 SPAI

println("\nBuilding Float64 CG-SPAI...")

M0_64 = spzeros(Float64, n, n)

M64, res64 =
    my_cg(A64, M0_64, spai_itmax, spai_tol)

I64 = spdiagm(0 => ones(Float64, n))

quality64 =
    norm(I64 - A64 * M64) / norm(I64)

spd64 =
    check_spd(M64; name = "CG-SPAI Float64")

# 4. Build Float32 SPAI

println("\nBuilding Float32 CG-SPAI...")

M0_32 = spzeros(Float32, n, n)

M32, res32 =
    my_cg(A32, M0_32, spai_itmax, spai_tol)

# Compute reported quality in Float64 for a fair comparison
M32_as64 = sparse(Float64.(M32))

quality32 =
    norm(I64 - A64 * M32_as64) / norm(I64)

spd32 =
    check_spd(M32_as64; name = "CG-SPAI Float32")

# 5. Plot residual convergence

p = plot(
    0:length(res64)-1,
    res64,
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "SPAI residual",
    title = "CG-SPAI precision comparison for HB/bcsstk21",
    label = "Float64",
    linewidth = 3
)

plot!(
    p,
    0:length(res32)-1,
    res32,
    label = "Float32",
    linestyle = :dash,
    linewidth = 2
)

savefig(
    p,
    "bcsstk21_cg_spai_float64_vs_float32.png"
)

println(
    "\nSaved plot: bcsstk21_cg_spai_float64_vs_float32.png"
)

common_length = min(length(res64), length(res32))

max_difference = maximum(
    abs.(res64[1:common_length] .- res32[1:common_length])
)

println("Maximum difference between residual histories: ", max_difference)

# 6. Print comparison table

println("\n=====")
println("SUMMARY")
println("=====")

println("\nFloat64")
println("SPAI iterations: ", length(res64) - 1)
println("SPAI quality: ", quality64)
println("Symmetry error: ", spd64.symmetry_error)
println("Smallest eigenvalue: ", spd64.lambda_min)
println("Cholesky succeeded: ", spd64.cholesky_success)
println("SPD: ", spd64.is_spd)
println("Nonzeros in M: ", nnz(M64))

println("\nFloat32")
println("SPAI iterations: ", length(res32) - 1)
println("SPAI quality evaluated in Float64: ", quality32)
println("Symmetry error: ", spd32.symmetry_error)
println("Smallest eigenvalue: ", spd32.lambda_min)
println("Cholesky succeeded: ", spd32.cholesky_success)
println("SPD: ", spd32.is_spd)
println("Nonzeros in M: ", nnz(M32))