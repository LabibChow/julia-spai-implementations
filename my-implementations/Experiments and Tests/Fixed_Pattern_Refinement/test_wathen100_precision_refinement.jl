using MatrixDepot
using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using Serialization

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")
float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision")
float16_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Float16")
utils_dir = joinpath(@__DIR__, "..", "..", "Utils")

include(joinpath(utils_dir, "frobenius.jl"))
include(joinpath(utils_dir, "dropping.jl"))
include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(float32_dir, "my_pmr_float32_spai.jl"))
include(joinpath(float32_dir, "my_pcg_float32_spai.jl"))
include(joinpath(float32_dir, "my_lopmr_float32_spai.jl"))
include(joinpath(float16_dir, "my_pmr_float16_spai.jl"))
include(joinpath(float16_dir, "my_pcg_float16_spai.jl"))
include(joinpath(float16_dir, "my_lopmr_float16_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))

iteration_endpoints = (5, 20, 100)
tol = 1.0e-15
max_per_column = 900
force_rerun_construction = false
force_rerun_refinement = false

itmax = first(iteration_endpoints)
run_tag = ""
results_dir = ""
matrix_dir = ""
checkpoint_dir = ""

function configure_endpoint!(endpoint::Int)
    endpoint ∈ iteration_endpoints || error("Invalid endpoint $endpoint. Expected one of $iteration_endpoints")

    global itmax = endpoint
    global run_tag = "it$(endpoint)_drop$(max_per_column)"
    global results_dir = joinpath(@__DIR__, "wathen100_precision_results", run_tag)
    global matrix_dir = joinpath(@__DIR__, "wathen100_saved_matrices", run_tag)
    global checkpoint_dir = joinpath(@__DIR__, "wathen100_checkpoints", run_tag)

    mkpath.((results_dir, matrix_dir, checkpoint_dir))
    println("\nCONFIGURED ENDPOINT: $endpoint ITERATIONS")
    println("Refinement will be applied to the low-precision pattern obtained at this endpoint.")
    return nothing
end

function relative_spai_residual(A::SparseMatrixCSC, M::SparseMatrixCSC)
    n = size(A, 1)
    A64 = eltype(A) == Float64 ? A : sparse(Float64.(A))
    M64 = eltype(M) == Float64 ? M : sparse(Float64.(M))
    R = spdiagm(0 => ones(Float64, n)) - A64 * M64
    result = frob_norm(R) / sqrt(Float64(n))
    R = nothing
    M64 = nothing
    GC.gc()
    return result
end

function pattern_is_subset(M_refined::SparseMatrixCSC, M_pattern::SparseMatrixCSC)
    size(M_refined) == size(M_pattern) || return false
    pattern_rows = rowvals(M_pattern)
    refined_rows = rowvals(M_refined)

    for j in axes(M_pattern, 2)
        allowed_rows = Set(pattern_rows[nzrange(M_pattern, j)])
        for pos in nzrange(M_refined, j)
            refined_rows[pos] ∈ allowed_rows || return false
        end
    end
    return true
end

maximum_column_nnz(M::SparseMatrixCSC) = maximum(diff(M.colptr))

function save_serialized(path, object)
    temp_path = path * ".tmp"
    serialize(temp_path, object)
    mv(temp_path, path; force=true)
    println("Saved: ", path)
    return nothing
end

function saved_matrix_path(method_name, precision_name; refined=false)
    suffix = refined ? "_refined" : ""
    return joinpath(matrix_dir, "$(lowercase(method_name))_$(lowercase(precision_name))$(suffix)_M.jls")
end

metadata_path(method_name, stage_name) = joinpath(checkpoint_dir, "$(lowercase(method_name))_$(stage_name).jls")
refinement_was_run(result) = result.refinement !== nothing
refined_residual_string(result) = refinement_was_run(result) ? @sprintf("%.6e", result.refinement.residual_after) : "SKIPPED"
refinement_time_string(result) = refinement_was_run(result) ? @sprintf("%.4f", result.refinement.refinement_time) : "SKIPPED"
refined_total_time_string(result) = refinement_was_run(result) ? @sprintf("%.4f", result.refinement.total_time) : "N/A"
refined_nnz_string(result) = refinement_was_run(result) ? string(result.refinement.nnz_after) : "SKIPPED"

function pattern_status_string(result)
    refinement_was_run(result) && return string(result.refinement.pattern_valid)
    return "SKIPPED — endpoint not reached ($(result.construction.iterations)/$itmax iterations)"
end

function construction_metadata(A, method_name, precision_name, residual_history, M, construction_time, M_path)
    nnz_M = nnz(M)
    return (
        method=method_name,
        precision=precision_name,
        iterations=length(residual_history) - 1,
        residual_history=residual_history,
        raw_residual=residual_history[end],
        residual=relative_spai_residual(A, M),
        nnz_M=nnz_M,
        density=nnz_M / length(M),
        max_column_nnz=maximum_column_nnz(M),
        symmetry_error=norm(M - transpose(M)),
        finite=all(isfinite, nonzeros(M)),
        construction_time=construction_time,
        matrix_path=M_path,
    )
end


function print_construction_stats(metadata)
    p = metadata.precision
    println("\n$p iterations:       ", metadata.iterations)
    @printf("%s raw residual:        %.6e\n", p, metadata.raw_residual)
    @printf("%s relative residual:   %.6e\n", p, metadata.residual)
    println("$p nnz(M):          ", metadata.nnz_M)
    @printf("%s density:             %.6e\n", p, metadata.density)
    println("$p max nnz/column:  ", metadata.max_column_nnz)
    @printf("%s symmetry error:      %.6e\n", p, metadata.symmetry_error)
    println("$p values finite:   ", metadata.finite)
    @printf("%s construction time:   %.4f seconds\n", p, metadata.construction_time)
end

function run_float64(method_name, spai_method, A, Pr, M0, itmax, tol, max_per_column)
    println("\n$method_name — FLOAT64")
    M_path = saved_matrix_path(method_name, "float64")
    meta_path = metadata_path(method_name, "float64")

    if !force_rerun_construction && isfile(M_path) && isfile(meta_path)
        println("Existing Float64 result found. Loading metadata only.")
        metadata = deserialize(meta_path)
        metadata.iterations == itmax || error("$method_name Float64 checkpoint has $(metadata.iterations) iterations, but endpoint $itmax was requested.")
        return metadata
    end

    println("Computing Float64 SPAI...")
    result = nothing
    construction_time = @elapsed begin
        result = spai_method(A, Pr, copy(M0), itmax, tol; max_per_column=max_per_column)
    end
    M64, residual_history = result
    result = nothing

    println("Saving Float64 matrix...")
    save_serialized(M_path, M64)

    metadata = construction_metadata(A, method_name, "Float64", residual_history, M64, construction_time, M_path)
    save_serialized(meta_path, metadata)
    print_construction_stats(metadata)

    M64 = nothing
    GC.gc()
    metadata.iterations == itmax || error("$method_name Float64 stopped after $(metadata.iterations) iterations; requested endpoint was $itmax.")
    return metadata
end

function construct_low_precision(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol, max_per_column)
    println("\n$method_name — $precision_name CONSTRUCTION")
    M_path = saved_matrix_path(method_name, precision_name)
    meta_path = metadata_path(method_name, lowercase(precision_name) * "_construction")

    if !force_rerun_construction && isfile(M_path) && isfile(meta_path)
        println("Existing $precision_name construction found. SPAI construction will NOT be repeated.")
        return deserialize(meta_path)
    end

    println("Computing $precision_name SPAI...")
    result = nothing
    construction_time = @elapsed begin
        result = spai_method(A, Pr, copy(M0), itmax, tol; max_per_column=max_per_column)
    end
    M_low, residual_history = result
    result = nothing

    println("Saving $precision_name SPAI before refinement...")
    save_serialized(M_path, M_low)

    metadata = construction_metadata(A, method_name, precision_name, residual_history, M_low, construction_time, M_path)
    save_serialized(meta_path, metadata)
    print_construction_stats(metadata)

    M_low = nothing
    GC.gc()
    return metadata
end

function refine_saved_pattern(method_name, precision_name, A, construction_metadata)
    println("\n$method_name — $precision_name + FLOAT64 REFINEMENT")
    refined_M_path = saved_matrix_path(method_name, precision_name; refined=true)
    refinement_meta_path = metadata_path(method_name, lowercase(precision_name) * "_refined")

    if !force_rerun_refinement && isfile(refined_M_path) && isfile(refinement_meta_path)
        println("Existing refined result found. Refinement will NOT be repeated.")
        return deserialize(refinement_meta_path)
    end

    low_M_path = construction_metadata.matrix_path
    isfile(low_M_path) || error("Saved low-precision matrix was not found:\n$low_M_path")

    println("Loading saved $precision_name SPAI:\n$low_M_path")
    M_low = deserialize(low_M_path)
    println("Loaded nnz(M_low): ", nnz(M_low))

    println("Starting Float64 fixed-pattern refinement...")
    M_refined = nothing
    refinement_time = @elapsed begin
        M_refined = refine_spai_pattern(A, M_low)
    end

    println("Saving refined matrix...")
    save_serialized(refined_M_path, M_refined)

    residual_before = construction_metadata.residual
    residual_after = relative_spai_residual(A, M_refined)
    improvement_factor = residual_after == 0.0 ? Inf : residual_before / residual_after
    nnz_after = nnz(M_refined)

    metadata = (
        method=method_name,
        precision=precision_name,
        residual_before=residual_before,
        residual_after=residual_after,
        improvement_factor=improvement_factor,
        nnz_before=construction_metadata.nnz_M,
        nnz_after=nnz_after,
        density_after=nnz_after / length(M_refined),
        max_column_nnz=maximum_column_nnz(M_refined),
        pattern_valid=pattern_is_subset(M_refined, M_low),
        symmetry_error=norm(M_refined - transpose(M_refined)),
        finite=all(isfinite, nonzeros(M_refined)),
        refinement_time=refinement_time,
        total_time=construction_metadata.construction_time + refinement_time,
        matrix_path=refined_M_path,
    )

    save_serialized(refinement_meta_path, metadata)
    @printf("\nResidual before refine:  %.6e\n", metadata.residual_before)
    @printf("Residual after refine:   %.6e\n", metadata.residual_after)
    @printf("Improvement factor:      %.6e\n", metadata.improvement_factor)
    println("nnz before:              ", metadata.nnz_before)
    println("nnz after:               ", metadata.nnz_after)
    println("Pattern valid:           ", metadata.pattern_valid)
    println("All values finite:       ", metadata.finite)
    @printf("Refinement time:         %.4f seconds\n", metadata.refinement_time)
    @printf("Construction + refine:   %.4f seconds\n", metadata.total_time)

    M_low = nothing
    M_refined = nothing
    GC.gc()
    return metadata
end

function run_low_precision_case(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol, max_per_column)
    construction = construct_low_precision(method_name, precision_name, spai_method, A, Pr, M0, itmax, tol, max_per_column)
    GC.gc()

    if construction.iterations != itmax
        println("\nENDPOINT NOT REACHED: $method_name $precision_name completed $(construction.iterations)/$itmax iterations.")
        if construction.iterations == 0
            println("No evolved $precision_name sparsity pattern was generated.")
        else
            println("A partial pattern exists, but it is not the requested iteration-$itmax endpoint pattern.")
        end
        println("Float64 fixed-pattern refinement will therefore be skipped for this endpoint.")
        GC.gc()
        return (construction=construction, refinement=nothing)
    end

    refinement = refine_saved_pattern(method_name, precision_name, A, construction)
    GC.gc()
    return (construction=construction, refinement=refinement)
end

function print_refined_residual(io, precision_name, result)
    if refinement_was_run(result)
        @printf(io, "%s refined residual: %.16e\n", precision_name, result.refinement.residual_after)
    else
        println(io, "$precision_name refined residual: SKIPPED — requested endpoint not reached")
    end
end

function print_refined_nnz(io, precision_name, result)
    refinement_was_run(result) ? println(io, "$precision_name refined nnz: ", result.refinement.nnz_after) : println(io, "$precision_name refined nnz: SKIPPED")
end

function print_refinement_times(io, precision_name, result)
    if refinement_was_run(result)
        @printf(io, "%s refinement time: %.6f\n", precision_name, result.refinement.refinement_time)
        @printf(io, "%s total time: %.6f\n", precision_name, result.refinement.total_time)
    else
        println(io, "$precision_name refinement time: SKIPPED")
        println(io, "$precision_name total time: N/A")
    end
end

function save_method_summary(method_name, baseline, result32, result16)
    slug = lowercase(method_name)
    path = joinpath(results_dir, "$(slug)_summary.txt")

    open(path, "w") do io
        println(io, "Matrix: wathen100")
        println(io, "Method: ", method_name)
        println(io, "itmax: ", itmax)
        println(io, "max_per_column: ", max_per_column, "\n")

        println(io, "Float64 iterations: ", baseline.iterations)
        println(io, "Float32 iterations: ", result32.construction.iterations)
        println(io, "Float16 iterations: ", result16.construction.iterations, "\n")

        @printf(io, "Float64 raw residual: %.16e\n", baseline.raw_residual)
        @printf(io, "Float64 relative residual: %.16e\n", baseline.residual)
        @printf(io, "Float32 raw residual: %.16e\n", result32.construction.raw_residual)
        @printf(io, "Float32 relative residual: %.16e\n", result32.construction.residual)
        print_refined_residual(io, "Float32", result32)
        @printf(io, "Float16 raw residual: %.16e\n", result16.construction.raw_residual)
        @printf(io, "Float16 relative residual: %.16e\n", result16.construction.residual)
        print_refined_residual(io, "Float16", result16)
        println(io)

        println(io, "Float64 nnz: ", baseline.nnz_M)
        println(io, "Float32 nnz: ", result32.construction.nnz_M)
        print_refined_nnz(io, "Float32", result32)
        println(io, "Float16 nnz: ", result16.construction.nnz_M)
        print_refined_nnz(io, "Float16", result16)
        println(io)

        @printf(io, "Float64 construction time: %.6f\n", baseline.construction_time)
        @printf(io, "Float32 construction time: %.6f\n", result32.construction.construction_time)
        print_refinement_times(io, "Float32", result32)
        @printf(io, "Float16 construction time: %.6f\n", result16.construction.construction_time)
        print_refinement_times(io, "Float16", result16)
        println(io)

        println(io, "Float32 pattern valid: ", pattern_status_string(result32))
        println(io, "Float16 pattern valid: ", pattern_status_string(result16))
    end

    writedlm(joinpath(results_dir, "$(slug)_float64_history.txt"), baseline.residual_history)
    writedlm(joinpath(results_dir, "$(slug)_float32_history.txt"), result32.construction.residual_history)
    writedlm(joinpath(results_dir, "$(slug)_float16_history.txt"), result16.construction.residual_history)

    scale = sqrt(30401.0)
    writedlm(joinpath(results_dir, "$(slug)_float64_relative_history.txt"), baseline.residual_history ./ scale)
    writedlm(joinpath(results_dir, "$(slug)_float32_relative_history.txt"), result32.construction.residual_history ./ scale)
    writedlm(joinpath(results_dir, "$(slug)_float16_relative_history.txt"), result16.construction.residual_history ./ scale)

    refined_rows = Matrix{Float64}(undef, 0, 2)
    if refinement_was_run(result32)
        refined_rows = vcat(refined_rows, [Float64(result32.construction.iterations) result32.refinement.residual_after])
    end
    if refinement_was_run(result16)
        refined_rows = vcat(refined_rows, [Float64(result16.construction.iterations) result16.refinement.residual_after])
    end
    writedlm(joinpath(results_dir, "$(slug)_refined_relative_endpoints.txt"), refined_rows)

    open(joinpath(results_dir, "$(slug)_refinement_status.txt"), "w") do io
        println(io, "Float32: ", refinement_was_run(result32) ? "COMPLETED" : "SKIPPED_ENDPOINT_NOT_REACHED")
        println(io, "Float16: ", refinement_was_run(result16) ? "COMPLETED" : "SKIPPED_ENDPOINT_NOT_REACHED")
    end

    println("\nSummary saved: ", path)
    return nothing
end

function print_run_summaries(results, methods)
    println("\nFINAL RELATIVE RESIDUAL SUMMARY")
    @printf("%-8s %-14s %-14s %-14s %-14s %-14s\n", "Method", "Float64", "Float32", "F32 refined", "Float16", "F16 refined")
    for name in methods
        baseline, result32, result16 = results[name]
        @printf("%-8s %-14.6e %-14.6e %-14s %-14.6e %-14s\n", name, baseline.residual,
            result32.construction.residual, refined_residual_string(result32),
            result16.construction.residual, refined_residual_string(result16))
    end

    println("\nRAW FROBENIUS RESIDUAL SUMMARY")
    @printf("%-8s %-18s %-18s %-18s\n", "Method", "Float64", "Float32", "Float16")
    for name in methods
        baseline, result32, result16 = results[name]
        @printf("%-8s %-18.6e %-18.6e %-18.6e\n", name, baseline.raw_residual,
            result32.construction.raw_residual, result16.construction.raw_residual)
    end

    println("\nITERATION SUMMARY")
    @printf("%-8s %-12s %-12s %-12s\n", "Method", "Float64", "Float32", "Float16")
    for name in methods
        baseline, result32, result16 = results[name]
        @printf("%-8s %-12d %-12d %-12d\n", name, baseline.iterations,
            result32.construction.iterations, result16.construction.iterations)
    end

    println("\nTIMING SUMMARY")
    @printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
        "Method", "F64", "F32", "F32 ref", "F32 total", "F16", "F16 ref", "F16 total")
    for name in methods
        baseline, result32, result16 = results[name]
        @printf("%-8s %-12.4f %-12.4f %-12s %-12s %-12.4f %-12s %-12s\n", name,
            baseline.construction_time, result32.construction.construction_time,
            refinement_time_string(result32), refined_total_time_string(result32),
            result16.construction.construction_time, refinement_time_string(result16),
            refined_total_time_string(result16))
    end

    println("\nSPARSITY SUMMARY")
    @printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "Method", "F64", "F32", "F32 ref", "F16", "F16 ref")
    for name in methods
        baseline, result32, result16 = results[name]
        @printf("%-8s %-12d %-12d %-12s %-12d %-12s\n", name, baseline.nnz_M,
            result32.construction.nnz_M, refined_nnz_string(result32),
            result16.construction.nnz_M, refined_nnz_string(result16))
    end

    println("\nFIXED-PATTERN CHECK")
    for name in methods
        _, result32, result16 = results[name]
        println(rpad("$name Float32:", 16), pattern_status_string(result32))
        println(rpad("$name Float16:", 16), pattern_status_string(result16))
    end
end

function main()
    println("\nwathen100 — PRECISION + FIXED-PATTERN REFINEMENT")

    matrix_name = "GHS_psdef/wathen100"
    println("Loading matrix from MatrixDepot...\n$matrix_name")
    md = mdopen(matrix_name)
    A = sparse(Float64.(md.A))
    md = nothing
    GC.gc()

    n = size(A, 1)
    println("\nMatrix information")
    println("Size:       ", size(A))
    println("nnz(A):     ", nnz(A))
    println("eltype(A):  ", eltype(A))
    @printf("Density(A): %.6e\n", nnz(A) / length(A))
    @printf("Symmetry error: %.6e\n", norm(A - transpose(A)))

    size(A) == (30401, 30401) || @warn "Unexpected dimensions." size=size(A)
    nnz(A) == 471601 || @warn "Unexpected nnz(A)." actual=nnz(A) expected=471601

    println("\nExperiment parameters")
    println("itmax:               ", itmax)
    println("tol:                 ", tol)
    println("max_per_column:      ", max_per_column)
    @printf("Nominal max density: %.6e\n", max_per_column / n)
    println("Run tag:             ", run_tag)

    A_diagonal = diag(A)
    any(iszero, A_diagonal) && error("Cannot form Jacobi preconditioner: A has a zero diagonal entry.")

    M0_64 = spdiagm(0 => ones(Float64, n))
    M0_32 = spdiagm(0 => ones(Float32, n))
    M0_16 = spdiagm(0 => ones(Float16, n))
    Pr64 = spdiagm(0 => 1.0 ./ A_diagonal)
    Pr32 = spdiagm(0 => Float32.(1.0 ./ A_diagonal))
    Pr16 = spdiagm(0 => Float16.(1.0 ./ A_diagonal))

    method_specs = (
        ("PMR", my_pmr, my_pmr_float32, my_pmr_float16),
        ("PCG", my_pcg, my_pcg_float32, my_pcg_float16),
        ("LOPMR", my_lopmr, my_lopmr_float32, my_lopmr_float16),
    )
    methods = first.(method_specs)
    results = Dict{String, Any}()

    for (name, method64, method32, method16) in method_specs
        println("\nSTARTING $name")
        baseline = run_float64(name, method64, A, Pr64, M0_64, itmax, tol, max_per_column)
        GC.gc()
        result32 = run_low_precision_case(name, "Float32", method32, A, Pr32, M0_32, itmax, tol, max_per_column)
        GC.gc()
        result16 = run_low_precision_case(name, "Float16", method16, A, Pr16, M0_16, itmax, tol, max_per_column)
        save_method_summary(name, baseline, result32, result16)
        results[name] = (baseline, result32, result16)
        GC.gc()
    end

    print_run_summaries(results, methods)
    println("\nwathen100 EXPERIMENT COMPLETE")
    println("Results:\n$results_dir")
    println("\nSaved matrices:\n$matrix_dir")
    println("\nCheckpoints:\n$checkpoint_dir")
end

function load_endpoint_metadata(endpoint, method_name)
    tag = "it$(endpoint)_drop$(max_per_column)"
    cdir = joinpath(@__DIR__, "wathen100_checkpoints", tag)
    slug = lowercase(method_name)

    f64_path = joinpath(cdir, "$(slug)_float64.jls")
    f32_path = joinpath(cdir, "$(slug)_float32_construction.jls")
    f32_ref_path = joinpath(cdir, "$(slug)_float32_refined.jls")
    f16_path = joinpath(cdir, "$(slug)_float16_construction.jls")
    f16_ref_path = joinpath(cdir, "$(slug)_float16_refined.jls")

    for path in (f64_path, f32_path, f16_path)
        isfile(path) || error("Missing endpoint metadata:\n$path")
    end

    return (
        f64=deserialize(f64_path),
        f32=deserialize(f32_path),
        f32_ref=isfile(f32_ref_path) ? deserialize(f32_ref_path) : nothing,
        f16=deserialize(f16_path),
        f16_ref=isfile(f16_ref_path) ? deserialize(f16_ref_path) : nothing,
    )
end

ref_value(ref, field; skipped="") = ref === nothing ? skipped : string(getproperty(ref, field))
ref_sci(ref, field; skipped="SKIPPED") = ref === nothing ? skipped : @sprintf("%.6e", getproperty(ref, field))

function write_combined_endpoint_summary()
    summary_root = joinpath(@__DIR__, "wathen100_precision_results")
    mkpath(summary_root)
    txt_path = joinpath(summary_root, "endpoint_summary_5_20_100.txt")
    csv_path = joinpath(summary_root, "endpoint_summary_5_20_100.csv")
    methods = ("PMR", "PCG", "LOPMR")

    open(txt_path, "w") do io
        println(io, "Matrix: wathen100")
        println(io, "SuiteSparse name: GHS_psdef/wathen100")
        println(io, "Dropping: max_per_column = ", max_per_column)
        println(io, "Endpoints: 5, 20, 100 iterations")
        println(io, "Refinement: Float64 fixed-pattern refinement performed separately after each endpoint\n")

        @printf(io, "%-8s %-6s %-14s %-14s %-14s %-14s %-14s %-12s %-12s %-12s %-12s %-12s\n",
            "Method", "Iter", "F64 relres", "F32 relres", "F32 refined", "F16 relres",
            "F16 refined", "F64 nnz", "F32 nnz", "F32 ref nnz", "F16 nnz", "F16 ref nnz")

        for endpoint in iteration_endpoints
            for method_name in methods
                data = load_endpoint_metadata(endpoint, method_name)
                @printf(io, "%-8s %-6d %-14.6e %-14.6e %-14s %-14.6e %-14s %-12d %-12d %-12s %-12d %-12s\n",
                    method_name, endpoint, data.f64.residual, data.f32.residual,
                    ref_sci(data.f32_ref, :residual_after), data.f16.residual,
                    ref_sci(data.f16_ref, :residual_after), data.f64.nnz_M, data.f32.nnz_M,
                    ref_value(data.f32_ref, :nnz_after; skipped="SKIPPED"), data.f16.nnz_M,
                    ref_value(data.f16_ref, :nnz_after; skipped="SKIPPED"))
            end
            println(io)
        end

        println(io, "\nNOTE: refinement is performed only when the corresponding low-precision construction reaches the requested endpoint exactly.")
    end

    headers = [
        "method", "endpoint", "f64_iterations", "f32_iterations", "f16_iterations",
        "f64_relative_residual", "f32_relative_residual", "f32_refined_relative_residual",
        "f16_relative_residual", "f16_refined_relative_residual", "f64_raw_residual",
        "f32_raw_residual", "f16_raw_residual", "f64_nnz", "f32_nnz", "f32_refined_nnz",
        "f16_nnz", "f16_refined_nnz", "f64_construction_time", "f32_construction_time",
        "f32_refinement_time", "f32_total_time", "f16_construction_time", "f16_refinement_time",
        "f16_total_time", "f32_pattern_valid", "f16_pattern_valid",
    ]

    open(csv_path, "w") do io
        println(io, join(headers, ","))
        for endpoint in iteration_endpoints, method_name in methods
            data = load_endpoint_metadata(endpoint, method_name)
            row = [
                method_name, string(endpoint), string(data.f64.iterations), string(data.f32.iterations), string(data.f16.iterations),
                string(data.f64.residual), string(data.f32.residual), ref_value(data.f32_ref, :residual_after),
                string(data.f16.residual), ref_value(data.f16_ref, :residual_after), string(data.f64.raw_residual),
                string(data.f32.raw_residual), string(data.f16.raw_residual), string(data.f64.nnz_M), string(data.f32.nnz_M),
                ref_value(data.f32_ref, :nnz_after), string(data.f16.nnz_M), ref_value(data.f16_ref, :nnz_after),
                string(data.f64.construction_time), string(data.f32.construction_time), ref_value(data.f32_ref, :refinement_time),
                ref_value(data.f32_ref, :total_time), string(data.f16.construction_time), ref_value(data.f16_ref, :refinement_time),
                ref_value(data.f16_ref, :total_time), ref_value(data.f32_ref, :pattern_valid; skipped="SKIPPED"),
                ref_value(data.f16_ref, :pattern_valid; skipped="SKIPPED"),
            ]
            println(io, join(row, ","))
        end
    end

    println("\nALL ENDPOINTS COMPLETE: 5, 20, 100")
    println("Combined text summary:\n$txt_path")
    println("\nCombined CSV summary:\n$csv_path")
    return nothing
end

println("\nwathen100 MULTI-ENDPOINT EXPERIMENT")
println("Endpoints: 5, 20, 100")
println("Each endpoint gets its OWN construction and refinement.")

for endpoint in iteration_endpoints
    configure_endpoint!(endpoint)
    main()
    GC.gc()
end

write_combined_endpoint_summary()