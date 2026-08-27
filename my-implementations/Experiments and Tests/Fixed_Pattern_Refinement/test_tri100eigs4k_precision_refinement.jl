using SparseArrays
using LinearAlgebra
using Random
using Printf
using DelimitedFiles
# Paths
spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)
float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)
float16_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Float16",)
utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)
# Output directories
history_dir = joinpath(@__DIR__, "tri100eigs4k_precision_histories",)
results_dir = joinpath(@__DIR__, "tri100eigs4k_precision_results",)
mkpath(history_dir)
mkpath(results_dir)
# Includes
include(joinpath(utils_dir,"frobenius.jl",))
# Float64 SPAI methods
include(joinpath(spai_dir, "my_pmr_spai.jl",))
include(joinpath(spai_dir, "my_pcg_spai.jl",))
include(joinpath(spai_dir, "my_lopmr_spai.jl",))
# Float32 SPAI methods
include(joinpath(float32_dir, "my_pmr_float32_spai.jl",))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl",))
include(joinpath(float32_dir, "my_lopmr_float32_spai.jl",))
# Float16 SPAI methods
include(joinpath(float16_dir, "my_pmr_float16_spai.jl",))
include(joinpath(float16_dir, "my_pcg_float16_spai.jl",))
include(joinpath(float16_dir, "my_lopmr_float16_spai.jl",))
# Fixed-pattern Float64 refinement
include(joinpath(spai_dir, "refine_spai_pattern.jl",))
# Helper functions
"""
Compute the relative SPAI residual
    ||I - A*M||_F / ||I||_F
using Float64 arithmetic.
"""
function relative_spai_residual(A::SparseMatrixCSC, M::SparseMatrixCSC,)
    n = size(A, 1)
    A64 = sparse(Float64.(A))
    M64 = sparse(Float64.(M))
    I64 = spdiagm(0 => ones(Float64, n,),)
    R = I64 - A64 * M64
    return frob_norm(R) / sqrt(Float64(n))
end
function pattern_is_subset(M_refined::SparseMatrixCSC, M_pattern::SparseMatrixCSC,)
    size(M_refined) == size(M_pattern) || return false
    pattern_rows = rowvals(M_pattern)
    refined_rows = rowvals(M_refined)
    for j in axes(M_pattern, 2)
        allowed_rows = Set(pattern_rows[nzrange(M_pattern, j,)],)
        for position in nzrange(M_refined, j,)
            refined_row = refined_rows[position]
            if refined_row ∉ allowed_rows
                return false
            end
        end
    end
    return true
end
"""
Verifies that PMR, PCG, and LOPMR histories begin with the same
residual for a given precision.
"""
function check_initial_residuals(precision_name, pmr_history, pcg_history, lopmr_history,)
    pmr_initial = Float64(pmr_history[1])
    pcg_initial = Float64(pcg_history[1])
    lopmr_initial = Float64(lopmr_history[1])
    println()
    println("$precision_name INITIAL RESIDUAL CHECK")
    @printf("PMR initial:   %.16e\n", pmr_initial,)
    @printf("PCG initial:   %.16e\n", pcg_initial,)
    @printf("LOPMR initial: %.16e\n", lopmr_initial,)
    consistent = isapprox(pmr_initial, pcg_initial; rtol = 1.0e-10, atol = 0.0,) && isapprox(pmr_initial, lopmr_initial;
            rtol = 1.0e-10, atol = 0.0,)
    if !consistent
        error("$precision_name initial residual histories are inconsistent. " * "Do not use or plot these histories.")
    end
    println()
    println("Initial residual check: PASSED")
    return nothing
end
# Float64 baseline
function test_float64(method_name::String, spai_method, A, Pr, M0, itmax, tol,)
    println()
    println("$method_name: Float64")
    result = nothing
    construction_time = @elapsed begin
            result = spai_method(A, Pr, copy(M0), itmax, tol,)
        end
    M64, residual_history = result
    relative_residual = relative_spai_residual(A, M64,)
    raw_residual = residual_history[end]
    println()
    println("Iterations:               ", length(residual_history) - 1,)
    @printf("Final raw residual:       %.6e\n", raw_residual,)
    @printf("Final relative residual:  %.6e\n", relative_residual,)
    println("nnz(M):                   ", nnz(M64),)
    @printf("Density of M:             %.6e\n", nnz(M64) / n^2,)
    @printf("Construction time:        %.4f seconds\n", construction_time,)
    return (method = method_name, M = M64, residual_history = residual_history, raw_residual = raw_residual, residual =
            relative_residual, nnz_M = nnz(M64), construction_time = construction_time,)
end
# Low precision + Float64 fixed-pattern refinement
function test_low_precision_refinement(method_name::String, precision_name::String, spai_method, A, Pr, M0, itmax, tol,)
    println()
    println("$method_name: $precision_name + Float64 refinement")
    # Low-precision SPAI construction
    println("Computing $precision_name SPAI...")
    result = nothing
    construction_time = @elapsed begin
            result = spai_method(A, Pr, copy(M0), itmax, tol,)
        end
    M_low, residual_history = result
    raw_residual_before = residual_history[end]
    residual_before = relative_spai_residual(A, M_low,)
    nnz_before = nnz(M_low)
    # Float64 fixed-pattern refinement
    println()
    println("Refining $precision_name sparsity pattern in Float64...")
    M_refined = nothing
    refinement_time = @elapsed begin
            M_refined = refine_spai_pattern(A, M_low,)
        end
    residual_after = relative_spai_residual(A, M_refined,)
    nnz_after = nnz(M_refined)
    improvement_factor = residual_after == 0.0 ? Inf : residual_before / residual_after
    pattern_valid = pattern_is_subset(M_refined, M_low,)
    # Output
    println()
    println("$precision_name iterations:        ", length(residual_history) - 1,)
    @printf("%s raw residual:          %.6e\n", precision_name, raw_residual_before,)
    @printf("%s relative residual:     %.6e\n", precision_name, residual_before,)
    @printf("Refined residual:         %.6e\n", residual_after,)
    @printf("Improvement factor:       %.6e\n", improvement_factor,)
    println()
    println("$precision_name nnz(M):           ", nnz_before,)
    println("Refined nnz(M):           ", nnz_after,)
    println("Pattern valid:            ", pattern_valid,)
    println()
    @printf("%s construction time:    %.4f seconds\n", precision_name, construction_time,)
    @printf("Float64 refinement time:  %.4f seconds\n", refinement_time,)
    @printf("Combined time:            %.4f seconds\n", construction_time + refinement_time,)
    return (method = method_name, precision = precision_name, M_low = M_low, M_refined = M_refined, residual_history =
            residual_history, raw_residual_before = raw_residual_before, residual_before = residual_before, residual_after =
            residual_after, improvement_factor = improvement_factor, nnz_before = nnz_before, nnz_after = nnz_after,
        pattern_valid = pattern_valid, construction_time = construction_time, refinement_time = refinement_time, total_time =
            construction_time + refinement_time,)
end
# Construct tri100eigs4k
Random.seed!(1)
n = 4000
k = 100
diagonal_entries = vcat(0.05 .+ rand(k), ones(n - k),)
subdiagonal_entries = rand(n - 1)
B = spdiagm(0 => diagonal_entries, -1 => subdiagonal_entries,)
A = sparse(B * transpose(B))
println()
println("Matrix: tri100eigs4k")
println("Size:   ", size(A),)
println("nnz(A): ", nnz(A),)
# Experiment parameters
itmax = 300
tol = 1.0e-15
println()
println("Experiment parameters")
println("itmax:     ", itmax,)
println("tol:       ", tol,)
println("dropping:  NONE",)
# Jacobi preconditioner
A_diagonal = diag(A)
if any(iszero, A_diagonal,)
    error("Jacobi preconditioner cannot be formed because " * "A contains a zero diagonal entry.")
end
# Initial matrices
M0_64 = spdiagm(0 => ones(Float64, n,),)
M0_32 = spdiagm(0 => ones(Float32, n,),)
M0_16 = spdiagm(0 => ones(Float16, n,),)
# Jacobi preconditioners
Pr64 = spdiagm(0 => 1.0 ./ A_diagonal,)
Pr32 = spdiagm(0 => Float32.(1.0 ./ A_diagonal),)
Pr16 = spdiagm(0 => Float16.(1.0 ./ A_diagonal),)
# FLOAT64
pmr_float64 = test_float64("PMR", my_pmr, A, Pr64, M0_64, itmax, tol,)
GC.gc()
pcg_float64 = test_float64("PCG", my_pcg, A, Pr64, M0_64, itmax, tol,)
GC.gc()
lopmr_float64 = test_float64("LOPMR", my_lopmr, A, Pr64, M0_64, itmax, tol,)
# Check Float64 iteration-0 histories
check_initial_residuals("Float64", pmr_float64.residual_history, pcg_float64.residual_history, lopmr_float64.residual_history,)
# FLOAT32 + FLOAT64 REFINEMENT
GC.gc()
pmr_float32 = test_low_precision_refinement("PMR", "Float32", my_pmr_float32, A, Pr32, M0_32, itmax, tol,)
GC.gc()
pcg_float32 = test_low_precision_refinement("PCG", "Float32", my_pcg_float32, A, Pr32, M0_32, itmax, tol,)
GC.gc()
lopmr_float32 = test_low_precision_refinement("LOPMR", "Float32", my_lopmr_float32, A, Pr32, M0_32, itmax, tol,)
# Check Float32 iteration-0 histories
check_initial_residuals("Float32", pmr_float32.residual_history, pcg_float32.residual_history, lopmr_float32.residual_history,)
# FLOAT16 + FLOAT64 REFINEMENT
GC.gc()
pmr_float16 = test_low_precision_refinement("PMR", "Float16", my_pmr_float16, A, Pr16, M0_16, itmax, tol,)
GC.gc()
pcg_float16 = test_low_precision_refinement("PCG", "Float16", my_pcg_float16, A, Pr16, M0_16, itmax, tol,)
GC.gc()
lopmr_float16 = test_low_precision_refinement("LOPMR", "Float16", my_lopmr_float16, A, Pr16, M0_16, itmax, tol,)
# Check Float16 iteration-0 histories
check_initial_residuals("Float16", pmr_float16.residual_history, pcg_float16.residual_history, lopmr_float16.residual_history,)
# Final residual summary
println()
println("FINAL RESIDUAL SUMMARY: tri100eigs4k")
@printf("%-8s %-14s %-14s %-14s %-14s %-14s\n", "Method", "Float64", "Float32", "F32 refined", "Float16", "F16 refined",)
for (baseline, result32, result16,) in ((pmr_float64, pmr_float32, pmr_float16,), (pcg_float64, pcg_float32, pcg_float16,),
    (lopmr_float64, lopmr_float32, lopmr_float16,),)
    @printf("%-8s %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e\n", baseline.method, baseline.residual, result32.residual_before,
        result32.residual_after, result16.residual_before, result16.residual_after,)
end
# Raw residual summary
println()
println("RAW FROBENIUS RESIDUAL SUMMARY")
@printf("%-8s %-18s %-18s %-18s\n", "Method", "Float64", "Float32", "Float16",)
for (baseline, result32, result16,) in ((pmr_float64, pmr_float32, pmr_float16,), (pcg_float64, pcg_float32, pcg_float16,),
    (lopmr_float64, lopmr_float32, lopmr_float16,),)
    @printf("%-8s %-18.6e %-18.6e %-18.6e\n", baseline.method, baseline.raw_residual, result32.raw_residual_before,
        result16.raw_residual_before,)
end
# Iteration summary
println()
println("ITERATION SUMMARY")
@printf("%-8s %-15s %-15s %-15s\n", "Method", "Float64", "Float32", "Float16",)
for (baseline, result32, result16,) in ((pmr_float64, pmr_float32, pmr_float16,), (pcg_float64, pcg_float32, pcg_float16,),
    (lopmr_float64, lopmr_float32, lopmr_float16,),)
    @printf("%-8s %-15d %-15d %-15d\n", baseline.method, length(baseline.residual_history) - 1, length(
            result32.residual_history) - 1, length(result16.residual_history) - 1,)
end
# Timing summary
println()
println("TIMING SUMMARY")
@printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", "Method", "F64", "F32", "F32 refine", "F32 total", "F16",
    "F16 refine", "F16 total",)
for (baseline, result32, result16,) in ((pmr_float64, pmr_float32, pmr_float16,), (pcg_float64, pcg_float32, pcg_float16,),
    (lopmr_float64, lopmr_float32, lopmr_float16,),)
    @printf("%-8s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n", baseline.method, baseline.construction_time,
        result32.construction_time, result32.refinement_time, result32.total_time, result16.construction_time,
        result16.refinement_time, result16.total_time,)
end
# Sparsity summary
println()
println("SPARSITY SUMMARY")
@printf("%-8s %-14s %-14s %-14s %-14s %-14s\n", "Method", "F64 nnz", "F32 nnz", "F32 ref nnz", "F16 nnz", "F16 ref nnz",)
for (baseline, result32, result16,) in ((pmr_float64, pmr_float32, pmr_float16,), (pcg_float64, pcg_float32, pcg_float16,),
    (lopmr_float64, lopmr_float32, lopmr_float16,),)
    @printf("%-8s %-14d %-14d %-14d %-14d %-14d\n", baseline.method, baseline.nnz_M, result32.nnz_before, result32.nnz_after,
        result16.nnz_before, result16.nnz_after,)
end
# Fixed-pattern checks
println()
println("FIXED-PATTERN CHECK")
println("PMR Float32 pattern valid:   ", pmr_float32.pattern_valid,)
println("PCG Float32 pattern valid:   ", pcg_float32.pattern_valid,)
println("LOPMR Float32 pattern valid: ", lopmr_float32.pattern_valid,)
println()
println("PMR Float16 pattern valid:   ", pmr_float16.pattern_valid,)
println("PCG Float16 pattern valid:   ", pcg_float16.pattern_valid,)
println("LOPMR Float16 pattern valid: ", lopmr_float16.pattern_valid,)
# Save residual histories
# These overwrite the old stale tri100eigs4k histories.
writedlm(joinpath(history_dir, "pmr_float64_history.txt",), pmr_float64.residual_history,)
writedlm(joinpath(history_dir, "pcg_float64_history.txt",), pcg_float64.residual_history,)
writedlm(joinpath(history_dir, "lopmr_float64_history.txt",), lopmr_float64.residual_history,)
writedlm(joinpath(history_dir, "pmr_float32_history.txt",), pmr_float32.residual_history,)
writedlm(joinpath(history_dir, "pcg_float32_history.txt",), pcg_float32.residual_history,)
writedlm(joinpath(history_dir, "lopmr_float32_history.txt",), lopmr_float32.residual_history,)
writedlm(joinpath(history_dir, "pmr_float16_history.txt",), pmr_float16.residual_history,)
writedlm(joinpath(history_dir, "pcg_float16_history.txt",), pcg_float16.residual_history,)
writedlm(joinpath(history_dir, "lopmr_float16_history.txt",), lopmr_float16.residual_history,)
# Save refined residuals
# column 1 = Float32 pattern + Float64 refinement
# column 2 = Float16 pattern + Float64 refinement
writedlm(joinpath(history_dir, "pmr_refined_residuals.txt",), [pmr_float32.residual_after pmr_float16.residual_after],)
writedlm(joinpath(history_dir, "pcg_refined_residuals.txt",), [pcg_float32.residual_after pcg_float16.residual_after],)
writedlm(joinpath(history_dir, "lopmr_refined_residuals.txt",), [lopmr_float32.residual_after lopmr_float16.residual_after],
)
# Save low-precision final relative residuals
# column 1 = Float32
# column 2 = Float16
writedlm(joinpath(history_dir, "pmr_low_precision_final_residuals.txt",), [pmr_float32.residual_before
        pmr_float16.residual_before],)
writedlm(joinpath(history_dir, "pcg_low_precision_final_residuals.txt",), [pcg_float32.residual_before
        pcg_float16.residual_before],)
writedlm(joinpath(history_dir, "lopmr_low_precision_final_residuals.txt",), [lopmr_float32.residual_before
        lopmr_float16.residual_before],)
# Save human-readable final summary
summary_file = joinpath(results_dir, "tri100eigs4k_precision_refinement_summary.txt",)
open(summary_file, "w",) do io
    println(io, "tri100eigs4k precision + fixed-pattern refinement")
    println(io, "Dropping: none")
    println(io, "itmax: ", itmax,)
    println(io, "tol: ", tol,)
    println(io)
    @printf(io, "%-8s %-14s %-14s %-14s %-14s %-14s\n", "Method", "Float64", "Float32", "F32 refined", "Float16", "F16 refined",
    )
    for (baseline, result32, result16,) in ((pmr_float64, pmr_float32, pmr_float16,), (pcg_float64, pcg_float32,
            pcg_float16,), (lopmr_float64, lopmr_float32, lopmr_float16,),)
        @printf(io, "%-8s %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e\n", baseline.method, baseline.residual,
            result32.residual_before, result32.residual_after, result16.residual_before, result16.residual_after,)
    end
end
# Finish
println()
println("TRI100EIGS4K EXPERIMENT COMPLETE")
println()
println("Histories saved to:")
println(history_dir)
println()
println("Summary saved to:")
println(summary_file)
