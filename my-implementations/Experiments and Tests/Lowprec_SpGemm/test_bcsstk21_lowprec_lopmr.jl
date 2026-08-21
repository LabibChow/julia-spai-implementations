using SparseArrays
using LinearAlgebra
using MatrixDepot
using Plots
using Printf
import Measures


include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "Utils", "dropping.jl"))

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_lopmr_spai.jl"))

include(joinpath(spai_dir, "my_lopmr_lowprec_spgemm.jl"))


# Load bcsstk21

md = mdopen("HB/bcsstk21")

A = sparse(Float64.(md.A))

n = size(A, 1)

println("Matrix: bcsstk21")
println("size(A) = $(size(A))")
println("nnz(A) = $(nnz(A))")
println("eltype(A) = $(eltype(A))")
println("density(A) = $(nnz(A) / n^2)")


# Initial SPAI and Jacobi preconditioner

M0 = spdiagm(
    0 => ones(Float64, n)
)

Pr = spdiagm(
    0 => 1.0 ./ diag(A)
)


# Experiment settings

itmax = 1000

tol = 1.0e-15

# Same symmetric dropping used for PMR and PCG.
max_per_column = 100

println("\nExperiment settings:")
println("itmax = $itmax")
println("tol = $tol")
println("max_per_column = $max_per_column")


# Warm-up runs

println("\nCompiling full-precision LOPMR...")

my_lopmr(
    A,
    Pr,
    copy(M0),
    1,
    tol;
    max_per_column = max_per_column
)

println("\nCompiling low-precision-SpGEMM LOPMR...")

my_lopmr_lowprec_spgemm(
    A,
    Pr,
    copy(M0),
    1,
    tol;
    max_per_column = max_per_column
)

GC.gc()


# Full Float64 LOPMR

println("\n")
println("FULL FLOAT64 LOPMR")
full64_result = nothing

full64_total_time = @elapsed begin

    full64_result = my_lopmr(
        A,
        Pr,
        copy(M0),
        itmax,
        tol;
        max_per_column = max_per_column
    )

end

M_full64, R_full64 = full64_result

# LOPMR with Float32 A*Z, A*P, and A*M

GC.gc()

println("\n")
println("LOPMR WITH FLOAT32 A*Z, A*P AND A*M SpGEMMs")

lowprec_result = nothing

lowprec_total_time = @elapsed begin

    lowprec_result = my_lopmr_lowprec_spgemm(
        A,
        Pr,
        copy(M0),
        itmax,
        tol;
        max_per_column = max_per_column
    )

end

M_lowprec,
R_lowprec,
lowprec_timings = lowprec_result


# Relative residual histories

relative_R_full64 =
    R_full64 ./ R_full64[1]

relative_R_lowprec =
    R_lowprec ./ R_lowprec[1]


I_sparse = spdiagm(
    0 => ones(Float64, n)
)

true_residual_full64 =
    frob_norm(
        I_sparse - A * M_full64
    ) /
    frob_norm(I_sparse)

true_residual_lowprec =
    frob_norm(
        I_sparse - A * M_lowprec
    ) /
    frob_norm(I_sparse)


# Symmetry diagnostics

symmetry_error_full64 =
    frob_norm(
        M_full64 -
        transpose(M_full64)
    ) /
    max(
        frob_norm(M_full64),
        eps(Float64)
    )

symmetry_error_lowprec =
    frob_norm(
        M_lowprec -
        transpose(M_lowprec)
    ) /
    max(
        frob_norm(M_lowprec),
        eps(Float64)
    )


# Experiment summary

println("\n")
println("EXPERIMENT SUMMARY")

@printf(
    "%-46s %15.6e\n",
    "Full Float64 final recorded residual:",
    R_full64[end]
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision final recorded residual:",
    R_lowprec[end]
)

@printf(
    "%-46s %15.6e\n",
    "Full Float64 true final residual:",
    true_residual_full64
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision true final residual:",
    true_residual_lowprec
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


# Low-precision SpGEMM timing breakdown

println("\n")
println("LOW-PRECISION SpGEMM TIMING BREAKDOWN")


@printf(
    "%-46s %15.6f s\n",
    "Z -> Float32 conversion time:",
    lowprec_timings.Z_to_Float32
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*Z SpGEMM time:",
    lowprec_timings.AZ_Float32_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "A*Z -> Float64 conversion time:",
    lowprec_timings.AZ_to_Float64
)


@printf(
    "%-46s %15.6f s\n",
    "P -> Float32 conversion time:",
    lowprec_timings.P_to_Float32
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*P SpGEMM time:",
    lowprec_timings.AP_Float32_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "A*P -> Float64 conversion time:",
    lowprec_timings.AP_to_Float64
)


@printf(
    "%-46s %15.6f s\n",
    "M -> Float32 conversion time:",
    lowprec_timings.M_to_Float32
)

@printf(
    "%-46s %15.6f s\n",
    "Float32 A*M SpGEMM time:",
    lowprec_timings.AM_Float32_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "A*M -> Float64 conversion time:",
    lowprec_timings.AM_to_Float64
)


# Float64 preconditioner products

@printf(
    "%-46s %15.6f s\n",
    "Float64 Pr*A*Z time:",
    lowprec_timings.PrAZ_Float64_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "Float64 Pr*A*P time:",
    lowprec_timings.PrAP_Float64_SpGEMM
)

# Totals

@printf(
    "%-46s %15.6f s\n",
    "Total conversion time:",
    lowprec_timings.total_conversion
)

@printf(
    "%-46s %15.6f s\n",
    "Total Float32 SpGEMM time:",
    lowprec_timings.total_Float32_SpGEMM
)

@printf(
    "%-46s %15.6f s\n",
    "Total Float64 preconditioner-product time:",
    lowprec_timings.total_preconditioner_SpGEMM
)


# Sparsity and symmetry

println("\n")
println("SPARSITY AND SYMMETRY")

@printf(
    "%-46s %15d\n",
    "nnz(M_full64):",
    nnz(M_full64)
)

@printf(
    "%-46s %15d\n",
    "nnz(M_lowprec):",
    nnz(M_lowprec)
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

@printf(
    "%-46s %15.6e\n",
    "Full Float64 symmetry error:",
    symmetry_error_full64
)

@printf(
    "%-46s %15.6e\n",
    "Low-precision symmetry error:",
    symmetry_error_lowprec
)


# Compare convergence histories

common_length = min(
    length(relative_R_full64),
    length(relative_R_lowprec)
)

maximum_history_difference =
    maximum(
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


# Convergence plot

plot(
    0:length(relative_R_full64)-1,
    relative_R_full64,
    yscale = :log10,
    linewidth = 2,
    label = "LOPMR Float64",
    xlabel = "Iteration",
    ylabel = "Relative residual norm",
    title = "bcsstk21: LOPMR Float64 vs Float32 SpGEMMs",
    legend = :topright,
    grid = true,
    margin = 5Measures.mm
)

plot!(
    0:length(relative_R_lowprec)-1,
    relative_R_lowprec,
    linewidth = 2,
    linestyle = :dash,
    label = "LOPMR Float32 A*Z, A*P and A*M"
)


# Save plot

output_path = joinpath(@__DIR__, "bcsstk21_lopmr_lowprec_spgemm.png"
)

savefig(output_path)

println("\nPlot saved to:")
println(output_path)