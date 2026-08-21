using MatrixDepot
using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using Serialization


# Paths
spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)
float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)
float16_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Float16",)
utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

# Output directories

history_dir = joinpath(@__DIR__, "msc04515_nodrop_precision_histories",)
checkpoint_dir = joinpath(@__DIR__, "msc04515_nodrop_checkpoints",)
results_dir = joinpath(@__DIR__, "msc04515_nodrop_precision_results",)

mkpath(history_dir)
mkpath(checkpoint_dir)
mkpath(results_dir)

# Includes
include(joinpath(utils_dir, "frobenius.jl",))

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

# Restart behaviour
# false: reuse existing checkpoints
# true: recompute everything from scratch

force_rerun = false

# Helper functions

function relative_spai_residual(
    A::SparseMatrixCSC,
    M::SparseMatrixCSC,
)
    n = size(A, 1)

    A64 = sparse(Float64.(A))
    M64 = sparse(Float64.(M))

    I64 = spdiagm(
        0 => ones(Float64, n),
    )

    R =
        I64 -
        A64 * M64

    return frob_norm(R) /
           sqrt(Float64(n))
end


function pattern_is_subset(
    M_refined::SparseMatrixCSC,
    M_pattern::SparseMatrixCSC,
)

    size(M_refined) ==
    size(M_pattern) ||
        return false

    pattern_rows =
        rowvals(M_pattern)

    refined_rows =
        rowvals(M_refined)

    for j in axes(M_pattern, 2)

        allowed_rows = Set(
            pattern_rows[
                nzrange(
                    M_pattern,
                    j,
                )
            ],
        )

        for position in nzrange(
            M_refined,
            j,
        )

            refined_row =
                refined_rows[position]

            if refined_row ∉ allowed_rows
                return false
            end
        end
    end

    return true
end

# Initial residual sanity check

function check_initial_residual(
    method_name,
    precision_name,
    history,
    expected_initial,
)

    actual_initial =
        Float64(history[1])


    if !isapprox(
        actual_initial,
        expected_initial;
        rtol = 1.0e-10,
        atol = 0.0,
    )

        error(
            "$method_name $precision_name initial residual is inconsistent.\n" *
            "Expected: $expected_initial\n" *
            "Actual:   $actual_initial"
        )

    end


    @printf(
        "%s %s initial residual check: PASSED (%.6e)\n",
        method_name,
        precision_name,
        actual_initial,
    )


    return nothing
end

# File-name helpers

function method_slug(
    method_name,
)
    return lowercase(method_name)
end


function precision_slug(
    precision_name,
)
    return lowercase(precision_name)
end


function construction_checkpoint_path(
    method_name,
    precision_name,
)

    return joinpath(
        checkpoint_dir,
        "$(method_slug(method_name))_$(precision_slug(precision_name))_construction.jls",
    )
end


function refinement_checkpoint_path(
    method_name,
    precision_name,
)

    return joinpath(
        checkpoint_dir,
        "$(method_slug(method_name))_$(precision_slug(precision_name))_refined.jls",
    )
end

# Serialization helpers
function save_checkpoint(
    path,
    result,
)

    serialize(
        path,
        result,
    )

    println()
    println(
        "Checkpoint saved:"
    )

    println(
        path
    )


    return nothing
end


function load_checkpoint(
    path,
)

    println()
    println("="^80)
    println("CHECKPOINT FOUND — LOADING")
    println(path)
    println("="^80)

    return deserialize(path)
end

# Save residual history immediately

function save_history(
    method_name,
    precision_name,
    residual_history,
)

    path = joinpath(
        history_dir,
        "$(method_slug(method_name))_$(precision_slug(precision_name))_history.txt",
    )


    writedlm(
        path,
        residual_history,
    )


    println(
        "History saved: ",
        path,
    )


    return path
end


# Construct or load one SPAI
# The checkpoint contains the actual final M matrix.

function run_or_load_construction(
    method_name,
    precision_name,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
    expected_initial,
)

    checkpoint =
        construction_checkpoint_path(
            method_name,
            precision_name,
        )

    # Reuse checkpoint

    if !force_rerun &&
       isfile(checkpoint)

        result =
            load_checkpoint(
                checkpoint
            )


        check_initial_residual(
            method_name,
            precision_name,
            result.residual_history,
            expected_initial,
        )


        save_history(
            method_name,
            precision_name,
            result.residual_history,
        )


        return result
    end

    # Fresh construction

    println()
    println("="^80)

    println(
        "$method_name — $precision_name — NO DROPPING"
    )

    println("="^80)


    raw_result = nothing


    construction_time =
        @elapsed begin

            raw_result =
                spai_method(
                    A,
                    Pr,
                    copy(M0),
                    itmax,
                    tol,
                )

        end


    M,
    residual_history =
        raw_result


    check_initial_residual(
        method_name,
        precision_name,
        residual_history,
        expected_initial,
    )


    raw_residual =
        residual_history[end]


    relative_residual =
        relative_spai_residual(
            A,
            M,
        )


    result = (
        method =
            method_name,

        precision =
            precision_name,

        # Final SPAI is saved for later spectrum analysis.
        M =
            M,

        residual_history =
            residual_history,

        raw_residual =
            raw_residual,

        relative_residual =
            relative_residual,

        nnz_M =
            nnz(M),

        construction_time =
            construction_time,
    )


    println()

    println(
        "Iterations:               ",
        length(residual_history) - 1,
    )

    @printf(
        "Final raw residual:       %.6e\n",
        raw_residual,
    )

    @printf(
        "Final relative residual:  %.6e\n",
        relative_residual,
    )

    println(
        "nnz(M):                   ",
        nnz(M),
    )

    @printf(
        "Density:                  %.6e\n",
        nnz(M) / size(A, 1)^2,
    )

    @printf(
        "Construction time:        %.4f seconds\n",
        construction_time,
    )

    # Save expensive construction FIRST.

    save_checkpoint(
        checkpoint,
        result,
    )


    save_history(
        method_name,
        precision_name,
        residual_history,
    )

    return result
end

# Refine or load a low-precision pattern

function run_or_load_refinement(
    method_name,
    precision_name,
    A,
    construction_result,
)

    checkpoint =
        refinement_checkpoint_path(
            method_name,
            precision_name,
        )

    # Load existing refined matrix

    if !force_rerun &&
       isfile(checkpoint)

        return load_checkpoint(
            checkpoint
        )

    end


    println()

    println(
        "Refining $method_name $precision_name pattern in Float64..."
    )


    M_refined = nothing

    refinement_time = @elapsed begin
    M_refined = refine_spai_pattern(
                    A,
                    construction_result.M,
                )
        end

    residual_after = relative_spai_residual(
            A,
            M_refined,
        )


    improvement_factor =
        residual_after == 0.0 ?
        Inf :
        construction_result.relative_residual /
        residual_after

    result = (

        M_refined =
            M_refined,

        residual_after =
            residual_after,

        improvement_factor =
            improvement_factor,

        nnz_after =
            nnz(M_refined),

        pattern_valid =
            pattern_is_subset(
                M_refined,
                construction_result.M,
            ),

        refinement_time =
            refinement_time,

        total_time =
            construction_result.construction_time +
            refinement_time,
    )


    @printf(
        "Before refinement:        %.6e\n",
        construction_result.relative_residual,
    )

    @printf(
        "After refinement:         %.6e\n",
        residual_after,
    )

    @printf(
        "Improvement factor:       %.6e\n",
        improvement_factor,
    )

    println(
        "Pattern valid:            ",
        result.pattern_valid,
    )

    println(
        "Refined nnz(M):           ",
        result.nnz_after,
    )

    @printf(
        "Refinement time:          %.4f seconds\n",
        refinement_time,
    )


    # Save refinement including M_refined.

    save_checkpoint(checkpoint, result,)

    return result
end


# These helpers deliberately remove the actual matrices so
# they can be released from RAM before the next long run.

function construction_metadata(
    result,
)

    return (
        method =
            result.method,

        precision =
            result.precision,

        iterations =
            length(result.residual_history) - 1,

        raw_residual =
            result.raw_residual,

        relative_residual =
            result.relative_residual,

        nnz_M =
            result.nnz_M,

        construction_time =
            result.construction_time,
    )
end


function refinement_metadata(
    result,
)

    return (
        residual_after =
            result.residual_after,

        improvement_factor =
            result.improvement_factor,

        nnz_after =
            result.nnz_after,

        pattern_valid =
            result.pattern_valid,

        refinement_time =
            result.refinement_time,

        total_time =
            result.total_time,
    )
end

# Save one method's lightweight summary

function save_method_summary(
    method_name,
    baseline,
    result32,
    refined32,
    result16,
    refined16,
)

    slug =
        method_slug(method_name)


    # Refined endpoint file used by plotting code.

    writedlm(
        joinpath(
            history_dir,
            "$(slug)_refined_residuals.txt",
        ),
        [
            refined32.residual_after
            refined16.residual_after
        ],
    )


    # Low-precision endpoint file.

    writedlm(
        joinpath(
            history_dir,
            "$(slug)_low_precision_final_residuals.txt",
        ),
        [
            result32.relative_residual
            result16.relative_residual
        ],
    )


    # Human-readable summary.

    summary_path =
        joinpath(
            results_dir,
            "$(slug)_summary.txt",
        )


    open(
        summary_path,
        "w",
    ) do io

        println(
            io,
            "msc04515 — $method_name — no dropping"
        )

        println(io)


        println(
            io,
            "Float64 iterations: ",
            baseline.iterations,
        )

        println(
            io,
            "Float32 iterations: ",
            result32.iterations,
        )

        println(
            io,
            "Float16 iterations: ",
            result16.iterations,
        )


        println(io)


        @printf(
            io,
            "Float64 raw residual: %.16e\n",
            baseline.raw_residual,
        )

        @printf(
            io,
            "Float64 relative residual: %.16e\n",
            baseline.relative_residual,
        )

        @printf(
            io,
            "Float32 raw residual: %.16e\n",
            result32.raw_residual,
        )

        @printf(
            io,
            "Float32 relative residual: %.16e\n",
            result32.relative_residual,
        )

        @printf(
            io,
            "Float32 refined residual: %.16e\n",
            refined32.residual_after,
        )

        @printf(
            io,
            "Float16 raw residual: %.16e\n",
            result16.raw_residual,
        )

        @printf(
            io,
            "Float16 relative residual: %.16e\n",
            result16.relative_residual,
        )

        @printf(
            io,
            "Float16 refined residual: %.16e\n",
            refined16.residual_after,
        )


        println(io)


        println(
            io,
            "Float64 nnz: ",
            baseline.nnz_M,
        )

        println(
            io,
            "Float32 nnz: ",
            result32.nnz_M,
        )

        println(
            io,
            "Float32 refined nnz: ",
            refined32.nnz_after,
        )

        println(
            io,
            "Float16 nnz: ",
            result16.nnz_M,
        )

        println(
            io,
            "Float16 refined nnz: ",
            refined16.nnz_after,
        )


        println(io)


        @printf(
            io,
            "Float64 construction time: %.6f\n",
            baseline.construction_time,
        )

        @printf(
            io,
            "Float32 construction time: %.6f\n",
            result32.construction_time,
        )

        @printf(
            io,
            "Float32 refinement time: %.6f\n",
            refined32.refinement_time,
        )

        @printf(
            io,
            "Float32 total time: %.6f\n",
            refined32.total_time,
        )

        @printf(
            io,
            "Float16 construction time: %.6f\n",
            result16.construction_time,
        )

        @printf(
            io,
            "Float16 refinement time: %.6f\n",
            refined16.refinement_time,
        )

        @printf(
            io,
            "Float16 total time: %.6f\n",
            refined16.total_time,
        )


        println(io)


        println(
            io,
            "Float32 pattern valid: ",
            refined32.pattern_valid,
        )

        println(
            io,
            "Float16 pattern valid: ",
            refined16.pattern_valid,
        )

    end


    println()
    println("="^80)

    println(
        "$method_name COMPLETE — ALL RESULTS CACHED"
    )

    println("="^80)


    return nothing
end


function run_method(
    method_name,
    method64,
    method32,
    method16,
    A,
    Pr64,
    Pr32,
    Pr16,
    M0_64,
    M0_32,
    M0_16,
    itmax,
    tol,
    expected_initial,
)

    println()
    println("#"^80)

    println(
        "STARTING $method_name"
    )

    println("#"^80)


    # Float64

    result64 =
        run_or_load_construction(
            method_name,
            "Float64",
            method64,
            A,
            Pr64,
            M0_64,
            itmax,
            tol,
            expected_initial,
        )


    meta64 =
        construction_metadata(
            result64
        )


    # Release large Float64 SPAI from RAM.
    result64 = nothing

    GC.gc()


    # Float32

    result32 =
        run_or_load_construction(
            method_name,
            "Float32",
            method32,
            A,
            Pr32,
            M0_32,
            itmax,
            tol,
            expected_initial,
        )


    refined32 =
        run_or_load_refinement(
            method_name,
            "Float32",
            A,
            result32,
        )


    meta32 =
        construction_metadata(
            result32
        )


    meta_refined32 =
        refinement_metadata(
            refined32
        )


    # Release both large matrices.
    result32 = nothing

    refined32 = nothing

    GC.gc()


    # Float16

    result16 =
        run_or_load_construction(
            method_name,
            "Float16",
            method16,
            A,
            Pr16,
            M0_16,
            itmax,
            tol,
            expected_initial,
        )


    refined16 =
        run_or_load_refinement(
            method_name,
            "Float16",
            A,
            result16,
        )


    meta16 =
        construction_metadata(
            result16
        )


    meta_refined16 =
        refinement_metadata(
            refined16
        )


    # Release both large matrices.
    result16 = nothing

    refined16 = nothing

    GC.gc()


    # Save lightweight method summary

    save_method_summary(
        method_name,
        meta64,
        meta32,
        meta_refined32,
        meta16,
        meta_refined16,
    )


    return (
        method =
            method_name,

        float64 =
            meta64,

        float32 =
            meta32,

        refined32 =
            meta_refined32,

        float16 =
            meta16,

        refined16 =
            meta_refined16,
    )
end


# Save spectrum/cache manifest

function save_cache_manifest(
    itmax,
    tol,
)

    manifest_path = joinpath(
        results_dir,
        "spectrum_cache_manifest.txt",
    )


    open(
        manifest_path,
        "w",
    ) do io

        println(
            io,
            "msc04515 no-dropping cached matrices"
        )

        println(
            io,
            "itmax = ",
            itmax,
        )

        println(
            io,
            "tol = ",
            tol,
        )

        println(
            io,
            "dropping = none"
        )

        println(io)

        println(
            io,
            "Original matrix:"
        )

        println(
            io,
            "msc04515_A_float64.jls"
        )

        println(io)


        for method in (
            "pmr",
            "pcg",
            "lopmr",
        )

            println(
                io,
                uppercase(method)
            )

            println(
                io,
                "  Float64 M: ",
                "$(method)_float64_construction.jls",
            )

            println(
                io,
                "  Float32 M: ",
                "$(method)_float32_construction.jls",
            )

            println(
                io,
                "  Float32 refined M: ",
                "$(method)_float32_refined.jls",
            )

            println(
                io,
                "  Float16 M: ",
                "$(method)_float16_construction.jls",
            )

            println(
                io,
                "  Float16 refined M: ",
                "$(method)_float16_refined.jls",
            )

            println(io)
        end

    end


    println(
        "Spectrum cache manifest saved to:"
    )

    println(
        manifest_path
    )


    return nothing
end


# Main experiment

function main()

    # Load msc04515

    md = mdopen("Boeing/msc04515",)

    A = sparse(Float64.(md.A),)

    n = size(A, 1)

    println()
    println("MSC04515 NO-DROPPING PRECISION + REFINEMENT")
    println("Size:   ", size(A),)
    println("nnz(A): ", nnz(A),)

    # Save A itself for later spectrum work.

    A_cache_path = joinpath(checkpoint_dir, "msc04515_A_float64.jls",)


    if force_rerun ||
       !isfile(A_cache_path)

        serialize(
            A_cache_path,
            A,
        )


        println()
        println(
            "Original A cached for spectrum analysis:"
        )

        println(
            A_cache_path
        )

    end

    # Experiment parameters

    itmax = 5000
    tol = 1.0e-15

    println()
    println("Experiment parameters")
    println("itmax:     ", itmax,)
    println("tol:       ", tol,)
    println("dropping:  NONE",)
    println("checkpoint reuse: ", !force_rerun,)

    # Jacobi preconditioner

    A_diagonal = diag(A)


    if any(iszero, A_diagonal,)
        error(
            "Jacobi preconditioner cannot be formed because " *
            "A contains a zero diagonal entry."
        )
    end

    # Initial SPAIs

    M0_64 = spdiagm(
            0 =>
                ones(
                    Float64,
                    n,
                ),
        )

    M0_32 = spdiagm(
            0 =>
                ones(
                    Float32,
                    n,
                ),
        )

    M0_16 =spdiagm(
            0 =>
                ones(
                    Float16,
                    n,
                ),
        )

    # Jacobi preconditioners

    Pr64 =
        spdiagm(
            0 =>
                1.0 ./
                A_diagonal,
        )


    Pr32 =
        spdiagm(
            0 =>
                Float32.(
                    1.0 ./
                    A_diagonal
                ),
        )


    Pr16 =
        spdiagm(
            0 =>
                Float16.(
                    1.0 ./
                    A_diagonal
                ),
        )


    # Expected iteration-0 residual

    I64 =
        spdiagm(
            0 =>
                ones(
                    Float64,
                    n,
                ),
        )


    expected_initial =
        frob_norm(
            I64 -
            A * M0_64
        )


    @printf(
        "Expected raw initial residual: %.16e\n",
        expected_initial,
    )


    # PMR
    pmr_summary =
        run_method(
            "PMR",
            my_pmr,
            my_pmr_float32,
            my_pmr_float16,
            A,
            Pr64,
            Pr32,
            Pr16,
            M0_64,
            M0_32,
            M0_16,
            itmax,
            tol,
            expected_initial,
        )


    GC.gc()

    # PCG
    pcg_summary =
        run_method(
            "PCG",
            my_pcg,
            my_pcg_float32,
            my_pcg_float16,
            A,
            Pr64,
            Pr32,
            Pr16,
            M0_64,
            M0_32,
            M0_16,
            itmax,
            tol,
            expected_initial,
        )

    GC.gc()

    # LOPMR
    lopmr_summary =
        run_method(
            "LOPMR",
            my_lopmr,
            my_lopmr_float32,
            my_lopmr_float16,
            A,
            Pr64,
            Pr32,
            Pr16,
            M0_64,
            M0_32,
            M0_16,
            itmax,
            tol,
            expected_initial,
        )

    GC.gc()

    # Final summary

    println()
    println("FINAL RESIDUAL SUMMARY: msc04515 — NO DROPPING")

    @printf(
        "%-8s %-14s %-14s %-14s %-14s %-14s\n",
        "Method",
        "Float64",
        "Float32",
        "F32 refined",
        "Float16",
        "F16 refined",
    )

    for summary in (
        pmr_summary,
        pcg_summary,
        lopmr_summary,
    )

        @printf(
            "%-8s %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e\n",
            summary.method,
            summary.float64.relative_residual,
            summary.float32.relative_residual,
            summary.refined32.residual_after,
            summary.float16.relative_residual,
            summary.refined16.residual_after,
        )

    end

    # Timing summary

    println()
    println("TIMING SUMMARY")

    @printf(
        "%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
        "Method",
        "F64",
        "F32",
        "F32 refine",
        "F32 total",
        "F16",
        "F16 refine",
        "F16 total",
    )


    for summary in (
        pmr_summary,
        pcg_summary,
        lopmr_summary,
    )

        @printf(
            "%-8s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
            summary.method,
            summary.float64.construction_time,
            summary.float32.construction_time,
            summary.refined32.refinement_time,
            summary.refined32.total_time,
            summary.float16.construction_time,
            summary.refined16.refinement_time,
            summary.refined16.total_time,
        )
    end

    # Sparsity summary

    println()
    println("SPARSITY SUMMARY")


    @printf(
        "%-8s %-14s %-14s %-14s %-14s %-14s\n",
        "Method",
        "F64 nnz",
        "F32 nnz",
        "F32 ref nnz",
        "F16 nnz",
        "F16 ref nnz",
    )

    for summary in (
        pmr_summary,
        pcg_summary,
        lopmr_summary,
    )

        @printf(
            "%-8s %-14d %-14d %-14d %-14d %-14d\n",
            summary.method,
            summary.float64.nnz_M,
            summary.float32.nnz_M,
            summary.refined32.nnz_after,
            summary.float16.nnz_M,
            summary.refined16.nnz_after,
        )
    end

    # Spectrum cache manifest

    save_cache_manifest(
        itmax,
        tol,
    )

    # Finish
    println()
    println("MSC04515 NO-DROPPING EXPERIMENT COMPLETE")
    println()
    println("Residual histories:")
    println(history_dir)

    println()
    println("Cached matrices for spectra:")
    println(checkpoint_dir)

    println()
    println("Text summaries:")

    println(results_dir)

    println()
    println("No SPAI construction needs to be rerun for the plots or spectra.")
end

main()