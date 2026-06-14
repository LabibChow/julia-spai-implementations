using MatrixDepot
using SparseArrays
using LinearAlgebra

include("../../Linear Solvers/my_cg_solve.jl")
include("../../Linear Solvers/my_pcg_solve.jl")
include("../../SPAI Iterations/my_cg_spai.jl")

# 1. Load matrix in Float64 for high-precision checking

matrix_name = "HB/bcsstk21"

md = mdopen(matrix_name)
A64 = sparse(Float64.(md.A))

n = size(A64, 1)

x_exact64 = ones(Float64, n)
b64 = A64 * x_exact64

# 2. Convert to Float32 for actual iterations

A32 = sparse(Float32.(A64))
b32 = Float32.(b64)
x032 = zeros(Float32, n)

itmax = 1000
tol = 1e-5

println("=====")
println("Matrix: ", matrix_name)
println("Size: ", size(A64))
println("Nonzeros in A: ", nnz(A64))
println("Precision: Float32 iteration, Float64 residual check")
println("=====")

# 3. CG in Float32, residual checked in Float64

println("\nRunning Float32 CG...")

x_cg32, res_cg32 = my_cg_solve(A32, b32, x032, itmax, tol)

cg32_rel_res64 = norm(b64 - A64 * Float64.(x_cg32)) / norm(b64)
cg32_rel_err64 = norm(Float64.(x_cg32) - x_exact64) / norm(x_exact64)

println("\nFloat32 CG Results")
println("Final relative residual checked in Float64: ", cg32_rel_res64)
println("Relative error checked in Float64: ", cg32_rel_err64)
println("Iterations: ", length(res_cg32) - 1)

# 4. PCG identity in Float32

M_identity32 = spdiagm(0 => ones(Float32, n))

println("\nRunning Float32 PCG with identity preconditioner...")

x_pcg_id32, res_pcg_id32 = my_pcg_solve(A32, b32, x032, M_identity32, itmax, tol)

pcg_id32_rel_res64 = norm(b64 - A64 * Float64.(x_pcg_id32)) / norm(b64)
pcg_id32_rel_err64 = norm(Float64.(x_pcg_id32) - x_exact64) / norm(x_exact64)

println("\nFloat32 PCG Identity Results")
println("Final relative residual checked in Float64: ", pcg_id32_rel_res64)
println("Relative error checked in Float64: ", pcg_id32_rel_err64)
println("Iterations: ", length(res_pcg_id32) - 1)

# 5. Build SPAI in Float32

println("\nBuilding Float32 SPAI preconditioner...")

spai_itmax = 75
spai_tol = 1e-3

M032 = spzeros(Float32, n, n)

M_spai32, spai_res32 = my_cg(A32, M032, spai_itmax, spai_tol)

println("\nFloat32 SPAI construction done.")
println("SPAI iterations: ", length(spai_res32) - 1)
println("Nonzeros in M_spai32: ", nnz(M_spai32))

# SPAI quality checked in Float64
M_spai64 = sparse(Float64.(M_spai32))
I64 = spdiagm(0 => ones(Float64, n))

R_spai64 = I64 - A64 * M_spai64
spai_quality64 = sqrt(sum(abs2, R_spai64)) / sqrt(sum(abs2, I64))

println("SPAI quality checked in Float64: ", spai_quality64)

# 6. PCG SPAI in Float32

println("\nRunning Float32 PCG with Float32 SPAI preconditioner...")

x_pcg_spai32, res_pcg_spai32 = my_pcg_solve(A32, b32, x032, M_spai32, itmax, tol)

pcg_spai32_rel_res64 = norm(b64 - A64 * Float64.(x_pcg_spai32)) / norm(b64)
pcg_spai32_rel_err64 = norm(Float64.(x_pcg_spai32) - x_exact64) / norm(x_exact64)

println("\nFloat32 PCG SPAI Results")
println("Final relative residual checked in Float64: ", pcg_spai32_rel_res64)
println("Relative error checked in Float64: ", pcg_spai32_rel_err64)
println("Iterations: ", length(res_pcg_spai32) - 1)

# 7. Summary

println("\n======")
println("FLOAT32 ITERATION + FLOAT64 RESIDUAL SUMMARY")
println("=====")

println("Matrix: ", matrix_name)
println("SPAI itmax: ", spai_itmax)

println("\nCG32 iterations: ", length(res_cg32) - 1)
println("PCG identity32 iterations: ", length(res_pcg_id32) - 1)
println("PCG SPAI32 iterations: ", length(res_pcg_spai32) - 1)

println("\nCG32 final residual checked in Float64: ", cg32_rel_res64)
println("PCG identity32 final residual checked in Float64: ", pcg_id32_rel_res64)
println("PCG SPAI32 final residual checked in Float64: ", pcg_spai32_rel_res64)

println("\nCG32 relative error checked in Float64: ", cg32_rel_err64)
println("PCG identity32 relative error checked in Float64: ", pcg_id32_rel_err64)
println("PCG SPAI32 relative error checked in Float64: ", pcg_spai32_rel_err64)

println("\nSPAI quality checked in Float64: ", spai_quality64)