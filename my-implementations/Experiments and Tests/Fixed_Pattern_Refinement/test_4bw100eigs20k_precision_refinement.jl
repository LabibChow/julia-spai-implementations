using MatrixMarket
using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using Serialization
# PATHS
spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)
float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)
float16_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Float16",)
utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)
# INCLUDES
include(joinpath(utils_dir, "frobenius.jl",))
include(joinpath(utils_dir, "dropping.jl",))
# Float64 implementations
include(joinpath(spai_dir, "my_pmr_spai.jl",))
include(joinpath(spai_dir, "my_pcg_spai.jl",))
include(joinpath(spai_dir, "my_lopmr_spai.jl",))
# Float32 implementations
include(joinpath(float32_dir, "my_pmr_float32_spai.jl",))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl",))
include(joinpath(float32_dir, "my_lopmr_float32_spai.jl",))
# Float16 implementations
include(joinpath(float16_dir, "my_pmr_float16_spai.jl",))
include(joinpath(float16_dir, "my_pcg_float16_spai.jl",))
include(joinpath(float16_dir, "my_lopmr_float16_spai.jl",))
# Float64 fixed-pattern refinement
include(joinpath(spai_dir, "refine_spai_pattern.jl",))
# SETTINGS
# This experiment runs THREE independent endpoint experiments:
#   5 iterations
#   20 iterations
#   100 iterations
#   like the working paper
# For EACH endpoint, the script constructs fresh Float64,
# Float32 and Float16 SPAIs from M0. Immediately after the
# Float32 / Float16 construction at that endpoint, it performs
# Float64 fixed-pattern refinement if the low-precision method
# completed at least one iteration.
iteration_endpoints = (5, 20, 100)
tol = 1.0e-15
max_per_column = 100
# Restart behaviour
# false: reuse completed construction/refinement results
# true: recompute them
force_rerun_construction = false
force_rerun_refinement = false
# ACTIVE ENDPOINT OUTPUT DIRECTORIES
# The resulting folders are:
#   it5_drop100
#   it20_drop100
#   it100_drop100
itmax = first(iteration_endpoints)
run_tag = ""
results_dir = ""
matrix_dir = ""
checkpoint_dir = ""
function configure_endpoint!(endpoint::Int)
    if endpoint ∉ iteration_endpoints
        error("Invalid endpoint $(endpoint). Expected one of " * string(iteration_endpoints))
    end
    global itmax = endpoint
    global run_tag = "it$(endpoint)_drop$(max_per_column)"
    global results_dir = joinpath(@__DIR__, "4bw100eigs20k_precision_results", run_tag,)
    global matrix_dir = joinpath(@__DIR__, "4bw100eigs20k_saved_matrices", run_tag,)
    global checkpoint_dir = joinpath(@__DIR__, "4bw100eigs20k_checkpoints", run_tag,)
    mkpath(results_dir)
    mkpath(matrix_dir)
    mkpath(checkpoint_dir)
    println()
    println("CONFIGURED ENDPOINT: $(endpoint) ITERATIONS")
    println("Refinement will be applied to the low-precision " * "pattern obtained at this endpoint.")
    return nothing
end
# Relative residual
# ||I - A*M||_F / ||I||_F
# ALWAYS evaluated in Float64
function relative_spai_residual(A::SparseMatrixCSC, M::SparseMatrixCSC,)
    n = size(A, 1)
    A64 = eltype(A) == Float64 ? A : sparse(Float64.(A))
    M64 = eltype(M) == Float64 ? M : sparse(Float64.(M))
    I64 = spdiagm(0 => ones(Float64, n,),)
    R = I64 - A64 * M64
    result = frob_norm(R) / sqrt(Float64(n))
    # Explicitly release large residual
    R = nothing
    M64 = nothing
    GC.gc()
    return result
end
# HELPER:
# Check refinement did not introduce locations outside the low-precision pattern
function pattern_is_subset(M_refined::SparseMatrixCSC, M_pattern::SparseMatrixCSC,)
    size(M_refined) == size(M_pattern) || return false
    pattern_rows = rowvals(M_pattern)
    refined_rows = rowvals(M_refined)
    for j in axes(M_pattern, 2,)
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
# HELPER:
# Maximum number of stored entries in any column
function maximum_column_nnz(M::SparseMatrixCSC,)
    return maximum(diff(M.colptr))
end
# HELPER:
# Save temporary file first, then rename
# Useful on COMA so an interrupted write is less likely
# to leave a fake "completed" checkpoint
function save_serialized(path, object,)
    temp_path = path * ".tmp"
    serialize(temp_path, object,)
    mv(temp_path, path; force = true,)
    println("Saved: ", path,)
    return nothing
end
# HELPER:
# Matrix filename
function saved_matrix_path(method_name, precision_name; refined = false,)
    method_slug = lowercase(method_name)
    precision_slug = lowercase(precision_name)
    suffix = refined ? "_refined" : ""
    return joinpath(matrix_dir, "$(method_slug)_$(precision_slug)$(suffix)_M.jls",)
end
# HELPER:
# Metadata checkpoint path
function metadata_path(method_name, stage_name,)
    method_slug = lowercase(method_name)
    return joinpath(checkpoint_dir, "$(method_slug)_$(stage_name).jls",)
end
# Refinement status / display
# A low-precision run that completes 0 iterations has not
# generated an evolved sparsity pattern. In that case the
# Float64 fixed-pattern refinement is intentionally skipped
refinement_was_run(result) = result.refinement !== nothing
function refined_residual_string(result,)
    if !refinement_was_run(result)
        return "SKIPPED"
    end
    return @sprintf("%.6e", result.refinement.residual_after,)
end
function refinement_time_string(result,)
    if !refinement_was_run(result)
        return "SKIPPED"
    end
    return @sprintf("%.4f", result.refinement.refinement_time,)
end
function refined_total_time_string(result,)
    if !refinement_was_run(result)
        return "N/A"
    end
    return @sprintf("%.4f", result.refinement.total_time,)
end
function refined_nnz_string(result,)
    if !refinement_was_run(result)
        return "SKIPPED"
    end
    return string(result.refinement.nnz_after)
end
function pattern_status_string(result,)
    if !refinement_was_run(result)
        return "SKIPPED — endpoint not reached (" * string(result.construction.iterations) * "/" * string(itmax) * " iterations)"
    end
    return string(result.refinement.pattern_valid)
end
# FLOAT64 BASELINE
# Saves M64 to disk.
function run_float64(method_name, spai_method, A, Pr, M0, itmax, tol, max_per_column,)
    println()
    println("$method_name — FLOAT64")
    M_path = saved_matrix_path(method_name, "float64",)
    meta_path = metadata_path(method_name, "float64",)
    # Reuse completed result
    if !force_rerun_construction && isfile(M_path) && isfile(meta_path)
        println()
        println("Existing Float64 result found.")
        println("Loading metadata only.")
        metadata = deserialize(meta_path)
        if metadata.iterations != itmax
            error("$method_name Float64 checkpoint has " *
                "$(metadata.iterations) iterations, but endpoint $(itmax) was requested.")
        end
        return metadata
    end
    # Construct SPAI
    println()
    println("Computing Float64 SPAI...")
    result = nothing
    construction_time = @elapsed begin
            result = spai_method(A, Pr, copy(M0), itmax, tol; max_per_column = max_per_column,)
        end
    M64, residual_history = result
    result = nothing
    # Save matrix IMMEDIATELY
    println()
    println("Saving Float64 matrix...")
    save_serialized(M_path, M64,)
    # Measurements
    raw_residual = residual_history[end]
    relative_residual = relative_spai_residual(A, M64,)
    nnz_M = nnz(M64)
    density = nnz_M / length(M64)
    max_col = maximum_column_nnz(M64)
    symmetry_error = norm(M64 - transpose(M64))
    finite = all(isfinite, nonzeros(M64),)
    metadata = (method = method_name, precision = "Float64", iterations = length(residual_history) - 1, residual_history =
            residual_history, raw_residual = raw_residual, residual = relative_residual, nnz_M = nnz_M, density = density,
        max_column_nnz = max_col, symmetry_error = symmetry_error, finite = finite, construction_time = construction_time,
        matrix_path = M_path,)
    save_serialized(meta_path, metadata,)
    # Output
    println()
    println("Iterations:              ", metadata.iterations,)
    @printf("Raw residual:            %.6e\n", raw_residual,)
    @printf("Relative residual:       %.6e\n", relative_residual,)
    println("nnz(M):                  ", nnz_M,)
    @printf("Density:                 %.6e\n", density,)
    println("Maximum nnz/column:      ", max_col,)
    @printf("Symmetry error:          %.6e\n", symmetry_error,)
    println("All values finite:       ", finite,)
    @printf("Construction time:       %.4f seconds\n", construction_time,)
    # RELEASE LARGE MATRIX
    M64 = nothing
    GC.gc()
    if metadata.iterations != itmax
        error("$method_name Float64 stopped after $(metadata.iterations) iterations; " * "the requested endpoint was $(itmax).")
    end
    return metadata
end
# LOW PRECISION CONSTRUCTION
# If the program crashes during refinement, the construction
# does NOT have to be repeated next time
function construct_low_precision(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol, max_per_column,)
    println()
    println("$method_name — $precision_name CONSTRUCTION")
    M_path = saved_matrix_path(method_name, precision_name,)
    stage_slug = lowercase(precision_name) * "_construction"
    meta_path = metadata_path(method_name, stage_slug,)
    # Reuse construction
    if !force_rerun_construction && isfile(M_path) && isfile(meta_path)
        println()
        println("Existing $precision_name construction found.")
        println("SPAI construction will NOT be repeated.")
        metadata = deserialize(meta_path)
        return metadata
    end
    # Construct
    println()
    println("Computing $precision_name SPAI...")
    result = nothing
    construction_time = @elapsed begin
            result = spai_method(A, Pr, copy(M0), itmax, tol; max_per_column = max_per_column,)
        end
    M_low, residual_history = result
    result = nothing
    # SAVE LOW-PRECISION MATRIX FIRST
    println()
    println("Saving $precision_name SPAI before refinement...")
    save_serialized(M_path, M_low,)
    # Measurements
    raw_residual = residual_history[end]
    relative_residual = relative_spai_residual(A, M_low,)
    nnz_M = nnz(M_low)
    density = nnz_M / length(M_low)
    max_col = maximum_column_nnz(M_low)
    symmetry_error = norm(M_low - transpose(M_low))
    finite = all(isfinite, nonzeros(M_low),)
    metadata = (method = method_name, precision = precision_name, iterations = length(residual_history) - 1, residual_history =
            residual_history, raw_residual = raw_residual, residual = relative_residual, nnz_M = nnz_M, density = density,
        max_column_nnz = max_col, symmetry_error = symmetry_error, finite = finite, construction_time = construction_time,
        matrix_path = M_path,)
    save_serialized(meta_path, metadata,)
    println()
    println("$precision_name iterations:       ", metadata.iterations,)
    @printf("%s raw residual:        %.6e\n", precision_name, raw_residual,)
    @printf("%s relative residual:   %.6e\n", precision_name, relative_residual,)
    println("$precision_name nnz(M):          ", nnz_M,)
    @printf("%s density:             %.6e\n", precision_name, density,)
    println("$precision_name max nnz/column:  ", max_col,)
    @printf("%s symmetry error:      %.6e\n", precision_name, symmetry_error,)
    println("$precision_name values finite:   ", finite,)
    @printf("%s construction time:   %.4f seconds\n", precision_name, construction_time,)
    #  M_low not returned
    # It is already stored on disk
    M_low = nothing
    GC.gc()
    return metadata
end
# FLOAT64 FIXED-PATTERN REFINEMENT
# Loads the saved low-precision M
function refine_saved_pattern(method_name, precision_name, A, construction_metadata,)
    println()
    println("$method_name — $precision_name + FLOAT64 REFINEMENT")
    refined_M_path = saved_matrix_path(method_name, precision_name; refined = true,)
    stage_slug = lowercase(precision_name) * "_refined"
    refinement_meta_path = metadata_path(method_name, stage_slug,)
    # Reuse completed refinement
    if !force_rerun_refinement && isfile(refined_M_path) && isfile(refinement_meta_path)
        println()
        println("Existing refined result found.")
        println("Refinement will NOT be repeated.")
        return deserialize(refinement_meta_path)
    end
    # Load the already saved low-precision SPAI
    low_M_path = construction_metadata.matrix_path
    if !isfile(low_M_path)
        error("Saved low-precision matrix was not found:\n" * low_M_path)
    end
    println()
    println("Loading saved $precision_name SPAI:")
    println(low_M_path)
    M_low = deserialize(low_M_path)
    println("Loaded nnz(M_low): ", nnz(M_low),)
    # Refinement
    println()
    println("Starting Float64 fixed-pattern refinement...")
    M_refined = nothing
    refinement_time = @elapsed begin
            M_refined = refine_spai_pattern(A, M_low,)
        end
    # SAVE REFINED MATRIX IMMEDIATELY
    println()
    println("Saving refined matrix...")
    save_serialized(refined_M_path,M_refined,)
    # Measurements
    residual_after = relative_spai_residual(A, M_refined,)
    residual_before = construction_metadata.residual
    improvement_factor = residual_after == 0.0 ? Inf : residual_before / residual_after
    nnz_after = nnz(M_refined)
    density_after = nnz_after / length(M_refined)
    pattern_valid = pattern_is_subset(M_refined, M_low,)
    max_col_after = maximum_column_nnz(M_refined)
    symmetry_error = norm(M_refined - transpose(M_refined))
    finite = all(isfinite, nonzeros(M_refined),)
    metadata = (method = method_name, precision = precision_name, residual_before = residual_before, residual_after =
            residual_after, improvement_factor = improvement_factor, nnz_before = construction_metadata.nnz_M, nnz_after =
            nnz_after, density_after = density_after, max_column_nnz = max_col_after, pattern_valid = pattern_valid,
        symmetry_error = symmetry_error, finite = finite, refinement_time = refinement_time, total_time =
            construction_metadata.construction_time + refinement_time, matrix_path = refined_M_path,)
    save_serialized(refinement_meta_path, metadata,)
    println()
    @printf("Residual before refine:  %.6e\n", residual_before,)
    @printf("Residual after refine:   %.6e\n", residual_after,)
    @printf("Improvement factor:      %.6e\n", improvement_factor,)
    println("nnz before:              ", construction_metadata.nnz_M,)
    println("nnz after:               ", nnz_after,)
    println("Pattern valid:           ", pattern_valid,)
    println("All values finite:       ", finite,)
    @printf("Refinement time:         %.4f seconds\n", refinement_time,)
    @printf("Construction + refine:   %.4f seconds\n", metadata.total_time,)
    # RELEASE BOTH LARGE MATRICES
    M_low = nothing
    M_refined = nothing
    GC.gc()
    return metadata
end
# RUN ONE LOW-PRECISION + REFINEMENT CASE
function run_low_precision_case(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol, max_per_column,)
    construction = construct_low_precision(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol, max_per_column,)
    GC.gc()
    # Exact-endpoint / expected-failure handling
    if construction.iterations != itmax
        println()
        println("ENDPOINT NOT REACHED: " * "$method_name $precision_name completed " *
            "$(construction.iterations) / $(itmax) iterations.")
        if construction.iterations == 0
            println("No evolved $precision_name sparsity pattern was generated.")
        else
            println("A partial pattern exists, but it is NOT the requested " * "iteration-$(itmax) endpoint pattern.")
        end
        println("Float64 fixed-pattern refinement will therefore be SKIPPED " * "for this endpoint.")
        GC.gc()
        return (construction = construction, refinement = nothing,)
    end
    # Refinement only when an evolved pattern exists
    refinement = refine_saved_pattern(method_name, precision_name, A, construction,)
    GC.gc()
    return (construction = construction, refinement = refinement,)
end
# SAVE HUMAN-READABLE RESULTS FOR ONE METHOD
function save_method_summary(method_name, baseline, result32, result16,)
    slug = lowercase(method_name)
    path = joinpath(results_dir, "$(slug)_summary.txt",)
    open(path, "w",) do io
        println(io, "Matrix: 4bw100eigs20k",)
        println(io, "Method: ", method_name,)
        println(io, "itmax: ", itmax,)
        println(io, "max_per_column: ", max_per_column,)
        println(io)
        # Iterations
        println(io, "Float64 iterations: ", baseline.iterations,)
        println(io, "Float32 iterations: ", result32.construction.iterations,)
        println(io, "Float16 iterations: ", result16.construction.iterations,)
        println(io)
        # Residuals
        @printf(io, "Float64 raw residual: %.16e\n", baseline.raw_residual,)
        @printf(io, "Float64 relative residual: %.16e\n", baseline.residual,)
        @printf(io, "Float32 raw residual: %.16e\n", result32.construction.raw_residual,)
        @printf(io, "Float32 relative residual: %.16e\n", result32.construction.residual,)
        if refinement_was_run(result32)
            @printf(io, "Float32 refined residual: %.16e\n", result32.refinement.residual_after,)
        else
            println(io, "Float32 refined residual: SKIPPED — requested endpoint not reached",)
        end
        @printf(io, "Float16 raw residual: %.16e\n", result16.construction.raw_residual,)
        @printf(io, "Float16 relative residual: %.16e\n", result16.construction.residual,)
        if refinement_was_run(result16)
            @printf(io, "Float16 refined residual: %.16e\n", result16.refinement.residual_after,)
        else
            println(io, "Float16 refined residual: SKIPPED — requested endpoint not reached",)
        end
        println(io)
        # Sparsity
        println(io, "Float64 nnz: ", baseline.nnz_M,)
        println(io, "Float32 nnz: ", result32.construction.nnz_M,)
        if refinement_was_run(result32)
            println(io, "Float32 refined nnz: ", result32.refinement.nnz_after,)
        else
            println(io, "Float32 refined nnz: SKIPPED",)
        end
        println(io, "Float16 nnz: ", result16.construction.nnz_M,)
        if refinement_was_run(result16)
            println(io, "Float16 refined nnz: ", result16.refinement.nnz_after,)
        else
            println(io, "Float16 refined nnz: SKIPPED",)
        end
        println(io)
        # Times
        @printf(io, "Float64 construction time: %.6f\n", baseline.construction_time,)
        @printf(io, "Float32 construction time: %.6f\n", result32.construction.construction_time,)
        if refinement_was_run(result32)
            @printf(io, "Float32 refinement time: %.6f\n", result32.refinement.refinement_time,)
            @printf(io, "Float32 total time: %.6f\n", result32.refinement.total_time,)
        else
            println(io, "Float32 refinement time: SKIPPED",)
            println(io, "Float32 total time: N/A",)
        end
        @printf(io, "Float16 construction time: %.6f\n", result16.construction.construction_time,)
        if refinement_was_run(result16)
            @printf(io, "Float16 refinement time: %.6f\n", result16.refinement.refinement_time,)
            @printf(io, "Float16 total time: %.6f\n", result16.refinement.total_time,)
        else
            println(io, "Float16 refinement time: SKIPPED",)
            println(io, "Float16 total time: N/A",)
        end
        println(io)
        # Pattern checks
        println(io, "Float32 pattern valid: ", pattern_status_string(result32),)
        println(io, "Float16 pattern valid: ", pattern_status_string(result16),)
    end
    # Raw residual histories
    writedlm(joinpath(results_dir, "$(slug)_float64_history.txt",), baseline.residual_history,)
    writedlm(joinpath(results_dir, "$(slug)_float32_history.txt",), result32.construction.residual_history,)
    writedlm(joinpath(results_dir, "$(slug)_float16_history.txt",), result16.construction.residual_history,)
    # Relative residual histories
    # raw residual / ||I||_F = raw residual / sqrt(n)
    n = 20000
    writedlm(joinpath(results_dir, "$(slug)_float64_relative_history.txt",), baseline.residual_history ./ sqrt(Float64(n)),)
    writedlm(joinpath(results_dir, "$(slug)_float32_relative_history.txt",), result32.construction.residual_history ./
        sqrt(Float64(n)),)
    writedlm(joinpath(results_dir, "$(slug)_float16_relative_history.txt",), result16.construction.residual_history ./
        sqrt(Float64(n)),)
    # Refined endpoints
    # Column 1 = iteration
    # Column 2 = relative residual after refinement
    refined_rows = Matrix{Float64}(undef, 0, 2,)
    if refinement_was_run(result32)
        refined_rows = vcat(refined_rows, [Float64(result32.construction.iterations) result32.refinement.residual_after],)
    end
    if refinement_was_run(result16)
        refined_rows = vcat(refined_rows, [Float64(result16.construction.iterations) result16.refinement.residual_after],)
    end
    writedlm(joinpath(results_dir, "$(slug)_refined_relative_endpoints.txt",), refined_rows,)
    # Explicit refinement status file
    open(joinpath(results_dir, "$(slug)_refinement_status.txt",), "w",) do io
        println(io, "Float32: ", refinement_was_run(result32) ? "COMPLETED" : "SKIPPED_ENDPOINT_NOT_REACHED",)
        println(io, "Float16: ", refinement_was_run(result16) ? "COMPLETED" : "SKIPPED_ENDPOINT_NOT_REACHED",)
    end
    println()
    println("Summary saved: ", path,)
    return nothing
end
# MAIN
function main()
    println()
    println("4bw100eigs20k — PRECISION + FIXED-PATTERN REFINEMENT")
    # LOAD MATRIX
    matrix_path = joinpath(@__DIR__, "4bw100eigs20k.mtx",)
    if !isfile(matrix_path)
        error("Matrix file not found:\n" * matrix_path)
    end
    println()
    println(matrix_path)
    A_raw = MatrixMarket.mmread(matrix_path)
    A = sparse(Float64.(sparse(A_raw)))
    A_raw = nothing
    GC.gc()
    n = size(A, 1,)
    println()
    println("Matrix information")
    println("Size:       ", size(A),)
    println("nnz(A):     ", nnz(A),)
    println("eltype(A):  ", eltype(A),)
    @printf("Density(A): %.6e\n", nnz(A) / length(A),)
    @printf("Symmetry error: %.6e\n", norm(A - transpose(A)),)
    if size(A) != (20000, 20000)
        @warn("Unexpected dimensions.", size = size(A),)
    end
    if nnz(A) != 179980
        @warn("Unexpected nnz(A).", actual = nnz(A), expected = 179980,)
    end
    # PARAMETERS
    println()
    println("Experiment parameters")
    println("itmax:               ", itmax,)
    println("tol:                 ", tol,)
    println("max_per_column:      ", max_per_column,)
    @printf("Nominal max density: %.6e\n", max_per_column / n,)
    println("Run tag:             ", run_tag,)
    # JACOBI PRECONDITIONER
    A_diagonal = diag(A)
    if any(iszero, A_diagonal,)
        error("Cannot form Jacobi preconditioner: " * "A has a zero diagonal entry.")
    end
    # Float64
    M0_64 = spdiagm(0 => ones(Float64, n,),)
    Pr64 = spdiagm(0 => 1.0 ./ A_diagonal,)
    # Float32
    M0_32 = spdiagm(0 => ones(Float32, n,),)
    Pr32 = spdiagm(0 => Float32.(1.0 ./ A_diagonal),)
    # Float16
    M0_16 = spdiagm(0 => ones(Float16, n,),)
    Pr16 = spdiagm(0 => Float16.(1.0 ./ A_diagonal),)
    # PMR
    println()
    println("STARTING PMR")
    pmr64 = run_float64("PMR", my_pmr, A, Pr64, M0_64, itmax, tol, max_per_column,)
    GC.gc()
    pmr32 = run_low_precision_case("PMR", "Float32", my_pmr_float32, A, Pr32, M0_32, itmax, tol, max_per_column,)
    GC.gc()
    pmr16 = run_low_precision_case("PMR", "Float16", my_pmr_float16, A, Pr16, M0_16, itmax, tol, max_per_column,)
    save_method_summary("PMR", pmr64, pmr32, pmr16,)
    GC.gc()
    # PCG
    println()
    println("STARTING PCG")
    pcg64 = run_float64("PCG", my_pcg, A, Pr64, M0_64, itmax, tol, max_per_column,)
    GC.gc()
    pcg32 = run_low_precision_case("PCG", "Float32", my_pcg_float32, A, Pr32, M0_32, itmax, tol, max_per_column,)
    GC.gc()
    pcg16 = run_low_precision_case("PCG", "Float16", my_pcg_float16, A, Pr16, M0_16, itmax, tol, max_per_column,)
    save_method_summary("PCG", pcg64, pcg32, pcg16,)
    GC.gc()
    # LOPMR
    println()
    println("STARTING LOPMR")
    lopmr64 = run_float64("LOPMR", my_lopmr, A, Pr64, M0_64, itmax, tol, max_per_column,)
    GC.gc()
    lopmr32 = run_low_precision_case("LOPMR", "Float32", my_lopmr_float32, A, Pr32, M0_32, itmax, tol, max_per_column,)
    GC.gc()
    lopmr16 = run_low_precision_case("LOPMR", "Float16", my_lopmr_float16, A, Pr16, M0_16, itmax, tol, max_per_column,)
    save_method_summary("LOPMR", lopmr64, lopmr32, lopmr16,)
    GC.gc()
    # FINAL RESIDUAL SUMMARY
    println()
    println("FINAL RELATIVE RESIDUAL SUMMARY")
    @printf("%-8s %-14s %-14s %-14s %-14s %-14s\n", "Method", "Float64", "Float32", "F32 refined", "Float16", "F16 refined",)
    for (baseline, result32, result16,) in ((pmr64, pmr32, pmr16,), (pcg64, pcg32, pcg16,), (lopmr64, lopmr32, lopmr16,),
    )
        @printf("%-8s %-14.6e %-14.6e %-14s %-14.6e %-14s\n", baseline.method, baseline.residual, result32.construction.residual,
            refined_residual_string(result32), result16.construction.residual, refined_residual_string(result16),)
    end
    # RAW RESIDUAL SUMMARY
    println()
    println("RAW FROBENIUS RESIDUAL SUMMARY")
    @printf("%-8s %-18s %-18s %-18s\n", "Method", "Float64", "Float32", "Float16",)
    for (baseline, result32, result16,) in ((pmr64, pmr32, pmr16,), (pcg64, pcg32, pcg16,), (lopmr64, lopmr32, lopmr16,),
    )
        @printf("%-8s %-18.6e %-18.6e %-18.6e\n", baseline.method, baseline.raw_residual, result32.construction.raw_residual,
            result16.construction.raw_residual,)
    end
    # ITERATION SUMMARY
    println()
    println("ITERATION SUMMARY")
    @printf("%-8s %-12s %-12s %-12s\n", "Method", "Float64", "Float32", "Float16",)
    for (baseline, result32, result16,) in ((pmr64, pmr32, pmr16,), (pcg64, pcg32, pcg16,), (lopmr64, lopmr32, lopmr16,),
    )
        @printf("%-8s %-12d %-12d %-12d\n", baseline.method, baseline.iterations, result32.construction.iterations,
            result16.construction.iterations,)
    end
    # TIMING SUMMARY
    println()
    println("TIMING SUMMARY")
    @printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", "Method", "F64", "F32", "F32 ref", "F32 total", "F16", "F16 ref",
        "F16 total",)
    for (baseline, result32, result16,) in ((pmr64, pmr32, pmr16,), (pcg64, pcg32, pcg16,), (lopmr64, lopmr32, lopmr16,),
    )
        @printf("%-8s %-12.4f %-12.4f %-12s %-12s %-12.4f %-12s %-12s\n", baseline.method, baseline.construction_time,
            result32.construction.construction_time, refinement_time_string(result32), refined_total_time_string(result32),
            result16.construction.construction_time, refinement_time_string(result16), refined_total_time_string(result16),)
    end
    # SPARSITY SUMMARY
    println()
    println("SPARSITY SUMMARY")
    @printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "Method", "F64", "F32", "F32 ref", "F16", "F16 ref",)
    for (baseline, result32, result16,) in ((pmr64, pmr32, pmr16,), (pcg64, pcg32, pcg16,), (lopmr64, lopmr32, lopmr16,),
    )
        @printf("%-8s %-12d %-12d %-12s %-12d %-12s\n", baseline.method, baseline.nnz_M, result32.construction.nnz_M,
            refined_nnz_string(result32), result16.construction.nnz_M, refined_nnz_string(result16),)
    end
    # PATTERN VALIDITY
    println()
    println("FIXED-PATTERN CHECK")
    println("PMR Float32:   ", pattern_status_string(pmr32),)
    println("PMR Float16:   ", pattern_status_string(pmr16),)
    println("PCG Float32:   ", pattern_status_string(pcg32),)
    println("PCG Float16:   ", pattern_status_string(pcg16),)
    println("LOPMR Float32: ", pattern_status_string(lopmr32),)
    println("LOPMR Float16: ", pattern_status_string(lopmr16),)
    # FINISH
    println()
    println("4bw100eigs20k EXPERIMENT COMPLETE")
    println()
    println("Results:")
    println(results_dir)
    println()
    println("Saved matrices:")
    println(matrix_dir)
    println()
    println("Checkpoints:")
    println(checkpoint_dir)
end
# COMBINED 5 / 20 / 100 ENDPOINT SUMMARY
function write_combined_endpoint_summary()
    summary_root = joinpath(@__DIR__, "4bw100eigs20k_precision_results",)
    mkpath(summary_root)
    txt_path = joinpath(summary_root, "endpoint_summary_5_20_100.txt",)
    csv_path = joinpath(summary_root, "endpoint_summary_5_20_100.csv",)
    methods = ("PMR", "PCG", "LOPMR")
    function load_endpoint_metadata(endpoint, method_name)
        tag = "it$(endpoint)_drop$(max_per_column)"
        cdir = joinpath(@__DIR__, "4bw100eigs20k_checkpoints", tag,)
        slug = lowercase(method_name)
        f64_path = joinpath(cdir, "$(slug)_float64.jls",)
        f32_path = joinpath(cdir, "$(slug)_float32_construction.jls",)
        f32_ref_path = joinpath(cdir, "$(slug)_float32_refined.jls",)
        f16_path = joinpath(cdir, "$(slug)_float16_construction.jls",)
        for required_path in (f64_path, f32_path, f16_path,)
            if !isfile(required_path)
                error("Missing endpoint metadata:\n" * required_path)
            end
        end
        f64 = deserialize(f64_path)
        f32 = deserialize(f32_path)
        f16 = deserialize(f16_path)
        f32_ref = isfile(f32_ref_path) ? deserialize(f32_ref_path) : nothing
        return (f64 = f64, f32 = f32, f32_ref = f32_ref, f16 = f16,)
    end
    open(txt_path, "w") do io
        println(io, "Matrix: 4bw100eigs20k")
        println(io, "Dropping: max_per_column = ", max_per_column)
        println(io, "Endpoints: 5, 20, 100 iterations")
        println(io, "Refinement: Float64 fixed-pattern refinement performed separately after each endpoint",)
        println(io)
        @printf(io, "%-8s %-6s %-14s %-14s %-14s %-14s %-12s %-12s %-12s\n", "Method", "Iter", "F64 relres", "F32 relres",
            "F32 refined", "F16 relres", "F64 nnz", "F32 nnz", "F32 ref nnz",)
        for endpoint in iteration_endpoints
            for method_name in methods
                data = load_endpoint_metadata(endpoint, method_name)
                f32_ref_string = data.f32_ref === nothing ? "SKIPPED" : @sprintf("%.6e", data.f32_ref.residual_after)
                f32_ref_nnz_string = data.f32_ref === nothing ? "SKIPPED" : string(data.f32_ref.nnz_after)
                @printf(io, "%-8s %-6d %-14.6e %-14.6e %-14s %-14.6e %-12d %-12d %-12s\n", method_name, endpoint,
                    data.f64.residual, data.f32.residual, f32_ref_string, data.f16.residual, data.f64.nnz_M, data.f32.nnz_M,
                    f32_ref_nnz_string,)
            end
            println(io)
        end
        println(io)
        println(io, "NOTE: low-precision refinement is performed only when the requested endpoint is reached exactly.",)
    end
    open(csv_path, "w") do io
        println(io, join(["method", "endpoint", "f64_iterations", "f32_iterations", "f16_iterations", "f64_relative_residual",
                    "f32_relative_residual", "f32_refined_relative_residual", "f16_relative_residual", "f64_raw_residual",
                    "f32_raw_residual", "f16_raw_residual", "f64_nnz", "f32_nnz", "f32_refined_nnz", "f16_nnz",
                    "f64_construction_time", "f32_construction_time", "f32_refinement_time", "f32_total_time",
                    "f16_construction_time", "f32_pattern_valid", "f16_refinement_status",], ",",))
        for endpoint in iteration_endpoints
            for method_name in methods
                data = load_endpoint_metadata(endpoint, method_name)
                f32_ref_residual = data.f32_ref === nothing ? "" : string(data.f32_ref.residual_after)
                f32_ref_nnz = data.f32_ref === nothing ? "" : string(data.f32_ref.nnz_after)
                f32_ref_time = data.f32_ref === nothing ? "" : string(data.f32_ref.refinement_time)
                f32_total_time = data.f32_ref === nothing ? "" : string(data.f32_ref.total_time)
                f32_pattern_valid = data.f32_ref === nothing ? "SKIPPED" : string(data.f32_ref.pattern_valid)
                f16_status = data.f16.iterations == endpoint ? "CONSTRUCTION_COMPLETED" : "SKIPPED_ENDPOINT_NOT_REACHED"
                row = [method_name, string(endpoint), string(data.f64.iterations), string(data.f32.iterations),
                    string(data.f16.iterations), string(data.f64.residual), string(data.f32.residual), f32_ref_residual,
                    string(data.f16.residual), string(data.f64.raw_residual), string(data.f32.raw_residual),
                    string(data.f16.raw_residual), string(data.f64.nnz_M), string(data.f32.nnz_M), f32_ref_nnz,
                    string(data.f16.nnz_M), string(data.f64.construction_time), string(data.f32.construction_time), f32_ref_time,
                    f32_total_time, string(data.f16.construction_time), f32_pattern_valid, f16_status,]
                println(io, join(row, ","))
            end
        end
    end
    println()
    println("ALL ENDPOINTS COMPLETE: 5, 20, 100")
    println()
    println("Combined text summary:")
    println(txt_path)
    println()
    println("Combined CSV summary:")
    println(csv_path)
    return nothing
end
# RUN EXACTLY THE THREE REQUESTED ENDPOINTS
println()
println("4bw100eigs20k MULTI-ENDPOINT EXPERIMENT")
println("Endpoints: 5, 20, 100")
println("Each endpoint gets its OWN construction and refinement.")
for endpoint in iteration_endpoints
    configure_endpoint!(endpoint)
    main()
    GC.gc()
end
write_combined_endpoint_summary()