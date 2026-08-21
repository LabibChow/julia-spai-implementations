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

# Includes
include(joinpath(utils_dir, "frobenius.jl",))

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

# Fixed-pattern Float64 refinement
include(joinpath(spai_dir, "refine_spai_pattern.jl",))

# Output directories

history_dir = joinpath(@__DIR__, "bcsstk21_nodrop_precision_histories",)

checkpoint_dir = joinpath(@__DIR__, "bcsstk21_nodrop_checkpoints",)

mkpath(history_dir)
mkpath(checkpoint_dir)

# Restart behaviour
# false: reuse a completed checkpoint if the script is restarted
# true: ignore all checkpoints and recompute everything

force_rerun = false


# Helper functions

"""
Compute the relative SPAI residual
    ||I - A*M||_F / ||I||_F

using Float64 arithmetic.
"""
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


"""
Check that fixed-pattern refinement introduced no stored
position outside the original low-precision pattern.
"""
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


# Float64 experiment

function test_float64(
    method_name::String,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)
    println()
    println("$method_name: Float64 — NO DROPPING")

    result = nothing

    construction_time = @elapsed begin

        result = spai_method(
                A,
                Pr,
                copy(M0),
                itmax,
                tol,
            )

    end


    M64,
    residual_history =
        result


    final_relative_residual =
        relative_spai_residual(
            A,
            M64,
        )

    final_raw_residual =
        residual_history[end]


    println()

    println(
        "Iterations:               ",
        length(residual_history) - 1,
    )

    @printf(
        "Final raw residual:       %.6e\n",
        final_raw_residual,
    )

    @printf(
        "Final relative residual:  %.6e\n",
        final_relative_residual,
    )

    println(
        "nnz(M):                   ",
        nnz(M64),
    )

    @printf(
        "Density of M:             %.6e\n",
        nnz(M64) / length(M64),
    )

    @printf(
        "Construction time:        %.4f seconds\n",
        construction_time,
    )


    return (
        method =
            method_name,

        M =
            M64,

        residual_history =
            residual_history,

        raw_residual =
            final_raw_residual,

        residual =
            final_relative_residual,

        nnz_M =
            nnz(M64),

        construction_time =
            construction_time,
    )
end


# Low precision + Float64 fixed-pattern refinement

function test_low_precision(
    method_name::String,
    precision_name::String,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)

    println()

    println("$method_name: $precision_name — NO DROPPING")


    # Low-precision SPAI construction

    println("Computing $precision_name SPAI...")

    result = nothing

    construction_time = @elapsed begin
        result =
            spai_method(
                A,
                Pr,
                copy(M0),
                itmax,
                tol,
            )
    end


    M_low,
    residual_history =
        result


    raw_residual_before =
        residual_history[end]


    residual_before =
        relative_spai_residual(
            A,
            M_low,
        )


    nnz_before =
        nnz(M_low)


    println()

    println(
        "$precision_name iterations: ",
        length(residual_history) - 1,
    )

    @printf(
        "%s raw residual:          %.6e\n",
        precision_name,
        raw_residual_before,
    )

    @printf(
        "%s relative residual:     %.6e\n",
        precision_name,
        residual_before,
    )

    println(
        "$precision_name nnz(M):   ",
        nnz_before,
    )

    @printf(
        "%s density:              %.6e\n",
        precision_name,
        nnz_before / length(M_low),
    )

    @printf(
        "%s construction time:    %.4f seconds\n",
        precision_name,
        construction_time,
    )


    # Float64 fixed-pattern refinement

    println()
    println("Refining $precision_name sparsity pattern in Float64...")


    M_refined = nothing

    refinement_time = @elapsed begin
        M_refined =
            refine_spai_pattern(
                A,
                M_low,
            )
    end


    residual_after =
        relative_spai_residual(
            A,
            M_refined,
        )


    nnz_after =
        nnz(M_refined)


    improvement_factor =
        residual_after == 0.0 ?
        Inf :
        residual_before /
        residual_after


    pattern_valid =
        pattern_is_subset(
            M_refined,
            M_low,
        )


    println()

    @printf(
        "Refined residual:         %.6e\n",
        residual_after,
    )

    @printf(
        "Improvement factor:       %.6e\n",
        improvement_factor,
    )

    println(
        "Refined nnz(M):           ",
        nnz_after,
    )

    println(
        "Pattern valid:            ",
        pattern_valid,
    )

    @printf(
        "Float64 refinement time:  %.4f seconds\n",
        refinement_time,
    )

    @printf(
        "Combined time:            %.4f seconds\n",
        construction_time +
        refinement_time,
    )


    return (
        method =
            method_name,

        precision =
            precision_name,

        M_low =
            M_low,

        M_refined =
            M_refined,

        residual_history =
            residual_history,

        raw_residual_before =
            raw_residual_before,

        residual_before =
            residual_before,

        residual_after =
            residual_after,

        improvement_factor =
            improvement_factor,

        nnz_before =
            nnz_before,

        nnz_after =
            nnz_after,

        pattern_valid =
            pattern_valid,

        construction_time =
            construction_time,

        refinement_time =
            refinement_time,

        total_time =
            construction_time +
            refinement_time,
    )
end


# Checkpoint helpers

function checkpoint_path(
    method_name::String,
    stage_name::String,
)

    method_slug =
        lowercase(method_name)

    return joinpath(
        checkpoint_dir,
        "$(method_slug)_$(stage_name).jls",
    )
end


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


function load_checkpoint(path)

    println()
    println("Existing checkpoint found.")
    println("Loading:")
    println(path)

    return deserialize(path)
end


# Run/load Float64

function run_or_load_float64(
    method_name,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)

    path =
        checkpoint_path(
            method_name,
            "float64_nodrop",
        )


    if !force_rerun &&
       isfile(path)

        return load_checkpoint(
            path
        )

    end


    result =
        test_float64(
            method_name,
            spai_method,
            A,
            Pr,
            M0,
            itmax,
            tol,
        )


    save_checkpoint(
        path,
        result,
    )


    return result
end


# Run/load low precision

function run_or_load_low_precision(
    method_name,
    precision_name,
    stage_name,
    spai_method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)

    path =
        checkpoint_path(
            method_name,
            stage_name,
        )


    if !force_rerun &&
       isfile(path)

        return load_checkpoint(
            path
        )

    end


    result =
        test_low_precision(
            method_name,
            precision_name,
            spai_method,
            A,
            Pr,
            M0,
            itmax,
            tol,
        )


    save_checkpoint(
        path,
        result,
    )


    return result
end


# Save results for one method immediately

function save_method_results(
    method_name,
    baseline,
    result32,
    result16,
)

    slug =
        lowercase(method_name)


    # Raw residual histories

    writedlm(
        joinpath(
            history_dir,
            "$(slug)_float64_history.txt",
        ),
        baseline.residual_history,
    )


    writedlm(
        joinpath(
            history_dir,
            "$(slug)_float32_history.txt",
        ),
        result32.residual_history,
    )


    writedlm(
        joinpath(
            history_dir,
            "$(slug)_float16_history.txt",
        ),
        result16.residual_history,
    )


    # Refined endpoint residuals
    # column 1 = Float32 pattern + Float64 refinement
    # column 2 = Float16 pattern + Float64 refinement

    writedlm(
        joinpath(
            history_dir,
            "$(slug)_refined_residuals.txt",
        ),
        [
            result32.residual_after
            result16.residual_after
        ],
    )


    # Low-precision endpoint residuals
    # column 1 = Float32 relative residual
    # column 2 = Float16 relative residual

    writedlm(
        joinpath(
            history_dir,
            "$(slug)_low_precision_final_residuals.txt",
        ),
        [
            result32.residual_before
            result16.residual_before
        ],
    )


    # Human-readable summary

    summary_path =
        joinpath(
            history_dir,
            "$(slug)_summary.txt",
        )


    open(
        summary_path,
        "w",
    ) do io

        println(
            io,
            "Method: ",
            method_name,
        )

        println(
            io,
            "Dropping: none",
        )

        println(io)


        println(
            io,
            "Float64 iterations: ",
            length(
                baseline.residual_history
            ) - 1,
        )

        println(
            io,
            "Float32 iterations: ",
            length(
                result32.residual_history
            ) - 1,
        )

        println(
            io,
            "Float16 iterations: ",
            length(
                result16.residual_history
            ) - 1,
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
            baseline.residual,
        )

        @printf(
            io,
            "Float32 raw residual: %.16e\n",
            result32.raw_residual_before,
        )

        @printf(
            io,
            "Float32 relative residual: %.16e\n",
            result32.residual_before,
        )

        @printf(
            io,
            "Float32 refined residual: %.16e\n",
            result32.residual_after,
        )

        @printf(
            io,
            "Float16 raw residual: %.16e\n",
            result16.raw_residual_before,
        )

        @printf(
            io,
            "Float16 relative residual: %.16e\n",
            result16.residual_before,
        )

        @printf(
            io,
            "Float16 refined residual: %.16e\n",
            result16.residual_after,
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
            result32.nnz_before,
        )

        println(
            io,
            "Float32 refined nnz: ",
            result32.nnz_after,
        )

        println(
            io,
            "Float16 nnz: ",
            result16.nnz_before,
        )

        println(
            io,
            "Float16 refined nnz: ",
            result16.nnz_after,
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
            result32.refinement_time,
        )

        @printf(
            io,
            "Float32 total time: %.6f\n",
            result32.total_time,
        )

        @printf(
            io,
            "Float16 construction time: %.6f\n",
            result16.construction_time,
        )

        @printf(
            io,
            "Float16 refinement time: %.6f\n",
            result16.refinement_time,
        )

        @printf(
            io,
            "Float16 total time: %.6f\n",
            result16.total_time,
        )


        println(io)


        println(
            io,
            "Float32 pattern valid: ",
            result32.pattern_valid,
        )

        println(
            io,
            "Float16 pattern valid: ",
            result16.pattern_valid,
        )

    end


    println()
    println("="^80)
    println("$method_name COMPLETE — RESULTS SAVED")
    println("="^80)

    println(
        summary_path
    )


    return nothing
end


# Main experiment

function main()

    # Load bcsstk21

    md =
        mdopen(
            "HB/bcsstk21",
        )


    A =
        sparse(
            Float64.(md.A),
        )


    n =
        size(A, 1)


    println()
    println("BCSSTK21 NO-DROPPING PRECISION + REFINEMENT")

    println(
        "Size:      ",
        size(A),
    )

    println(
        "nnz(A):    ",
        nnz(A),
    )


    # Experiment parameters

    itmax = 1000

    tol = 1.0e-15


    println()
    println("Experiment parameters")

    println(
        "itmax:     ",
        itmax,
    )

    println(
        "tol:       ",
        tol,
    )

    println(
        "dropping:  NONE",
    )


    # --------------------------------------------------------
    # Jacobi preconditioner
    # --------------------------------------------------------

    A_diagonal =
        diag(A)


    if any(
        iszero,
        A_diagonal,
    )

        error(
            "Jacobi preconditioner cannot be formed because " *
            "A contains a zero diagonal entry."
        )

    end


    # --------------------------------------------------------
    # Float64 inputs
    # --------------------------------------------------------

    M0_64 =
        spdiagm(
            0 =>
                ones(
                    Float64,
                    n,
                ),
        )


    Pr64 =
        spdiagm(
            0 =>
                1.0 ./
                A_diagonal,
        )


    # Float32 inputs

    M0_32 =
        spdiagm(
            0 =>
                ones(
                    Float32,
                    n,
                ),
        )


    Pr32 =
        spdiagm(
            0 =>
                Float32.(
                    1.0 ./
                    A_diagonal
                ),
        )


    # Float16 inputs

    M0_16 =
        spdiagm(
            0 =>
                ones(
                    Float16,
                    n,
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


    # PMR

    println()
    println("STARTING PMR")


    pmr_float64 =
        run_or_load_float64(
            "PMR",
            my_pmr,
            A,
            Pr64,
            M0_64,
            itmax,
            tol,
        )


    GC.gc()


    pmr_float32 =
        run_or_load_low_precision(
            "PMR",
            "Float32",
            "float32_nodrop_refined",
            my_pmr_float32,
            A,
            Pr32,
            M0_32,
            itmax,
            tol,
        )


    GC.gc()


    pmr_float16 =
        run_or_load_low_precision(
            "PMR",
            "Float16",
            "float16_nodrop_refined",
            my_pmr_float16,
            A,
            Pr16,
            M0_16,
            itmax,
            tol,
        )


    save_method_results(
        "PMR",
        pmr_float64,
        pmr_float32,
        pmr_float16,
    )


    GC.gc()


    # PCG
    println()
    println("STARTING PCG")


    pcg_float64 =
        run_or_load_float64(
            "PCG",
            my_pcg,
            A,
            Pr64,
            M0_64,
            itmax,
            tol,
        )


    GC.gc()


    pcg_float32 =
        run_or_load_low_precision(
            "PCG",
            "Float32",
            "float32_nodrop_refined",
            my_pcg_float32,
            A,
            Pr32,
            M0_32,
            itmax,
            tol,
        )


    GC.gc()


    pcg_float16 =
        run_or_load_low_precision(
            "PCG",
            "Float16",
            "float16_nodrop_refined",
            my_pcg_float16,
            A,
            Pr16,
            M0_16,
            itmax,
            tol,
        )


    save_method_results(
        "PCG",
        pcg_float64,
        pcg_float32,
        pcg_float16,
    )


    GC.gc()


    # LOPMR

    println()
    println("STARTING LOPMR")

    lopmr_float64 =
        run_or_load_float64(
            "LOPMR",
            my_lopmr,
            A,
            Pr64,
            M0_64,
            itmax,
            tol,
        )


    GC.gc()


    lopmr_float32 =
        run_or_load_low_precision(
            "LOPMR",
            "Float32",
            "float32_nodrop_refined",
            my_lopmr_float32,
            A,
            Pr32,
            M0_32,
            itmax,
            tol,
        )


    GC.gc()


    lopmr_float16 =
        run_or_load_low_precision(
            "LOPMR",
            "Float16",
            "float16_nodrop_refined",
            my_lopmr_float16,
            A,
            Pr16,
            M0_16,
            itmax,
            tol,
        )


    save_method_results(
        "LOPMR",
        lopmr_float64,
        lopmr_float32,
        lopmr_float16,
    )


    GC.gc()


    # Final residual summary

    println()
    println("FINAL RESIDUAL SUMMARY: bcsstk21 — NO DROPPING")


    @printf(
        "%-8s %-14s %-14s %-14s %-14s %-14s\n",
        "Method",
        "Float64",
        "Float32",
        "F32 refined",
        "Float16",
        "F16 refined",
    )


    for (
        baseline,
        result32,
        result16,
    ) in (
        (
            pmr_float64,
            pmr_float32,
            pmr_float16,
        ),
        (
            pcg_float64,
            pcg_float32,
            pcg_float16,
        ),
        (
            lopmr_float64,
            lopmr_float32,
            lopmr_float16,
        ),
    )

        @printf(
            "%-8s %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e\n",
            baseline.method,
            baseline.residual,
            result32.residual_before,
            result32.residual_after,
            result16.residual_before,
            result16.residual_after,
        )

    end


    # Raw residual summary

    println()
    println("RAW FROBENIUS RESIDUAL SUMMARY")


    @printf(
        "%-8s %-18s %-18s %-18s\n",
        "Method",
        "Float64",
        "Float32",
        "Float16",
    )


    for (
        baseline,
        result32,
        result16,
    ) in (
        (
            pmr_float64,
            pmr_float32,
            pmr_float16,
        ),
        (
            pcg_float64,
            pcg_float32,
            pcg_float16,
        ),
        (
            lopmr_float64,
            lopmr_float32,
            lopmr_float16,
        ),
    )

        @printf(
            "%-8s %-18.6e %-18.6e %-18.6e\n",
            baseline.method,
            baseline.raw_residual,
            result32.raw_residual_before,
            result16.raw_residual_before,
        )

    end


    # Iteration summary

    println()
    println("ITERATION SUMMARY")


    @printf(
        "%-8s %-15s %-15s %-15s\n",
        "Method",
        "Float64",
        "Float32",
        "Float16",
    )


    for (
        baseline,
        result32,
        result16,
    ) in (
        (
            pmr_float64,
            pmr_float32,
            pmr_float16,
        ),
        (
            pcg_float64,
            pcg_float32,
            pcg_float16,
        ),
        (
            lopmr_float64,
            lopmr_float32,
            lopmr_float16,
        ),
    )

        @printf(
            "%-8s %-15d %-15d %-15d\n",
            baseline.method,
            length(
                baseline.residual_history
            ) - 1,
            length(
                result32.residual_history
            ) - 1,
            length(
                result16.residual_history
            ) - 1,
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


    for (
        baseline,
        result32,
        result16,
    ) in (
        (
            pmr_float64,
            pmr_float32,
            pmr_float16,
        ),
        (
            pcg_float64,
            pcg_float32,
            pcg_float16,
        ),
        (
            lopmr_float64,
            lopmr_float32,
            lopmr_float16,
        ),
    )

        @printf(
            "%-8s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
            baseline.method,
            baseline.construction_time,
            result32.construction_time,
            result32.refinement_time,
            result32.total_time,
            result16.construction_time,
            result16.refinement_time,
            result16.total_time,
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


    for (
        baseline,
        result32,
        result16,
    ) in (
        (
            pmr_float64,
            pmr_float32,
            pmr_float16,
        ),
        (
            pcg_float64,
            pcg_float32,
            pcg_float16,
        ),
        (
            lopmr_float64,
            lopmr_float32,
            lopmr_float16,
        ),
    )

        @printf(
            "%-8s %-14d %-14d %-14d %-14d %-14d\n",
            baseline.method,
            baseline.nnz_M,
            result32.nnz_before,
            result32.nnz_after,
            result16.nnz_before,
            result16.nnz_after,
        )

    end


    # Fixed-pattern validity

    println()
    println("FIXED-PATTERN CHECK")


    println(
        "PMR Float32 pattern valid:   ",
        pmr_float32.pattern_valid,
    )

    println(
        "PCG Float32 pattern valid:   ",
        pcg_float32.pattern_valid,
    )

    println(
        "LOPMR Float32 pattern valid: ",
        lopmr_float32.pattern_valid,
    )

    println()

    println(
        "PMR Float16 pattern valid:   ",
        pmr_float16.pattern_valid,
    )

    println(
        "PCG Float16 pattern valid:   ",
        pcg_float16.pattern_valid,
    )

    println(
        "LOPMR Float16 pattern valid: ",
        lopmr_float16.pattern_valid,
    )

    # Finish

    println()
    println("BCSSTK21 NO-DROPPING EXPERIMENT COMPLETE")

    println()
    println(
        "Histories/results:"
    )

    println(
        history_dir
    )

    println()
    println(
        "Checkpoints:"
    )

    println(
        checkpoint_dir
    )

end

main()