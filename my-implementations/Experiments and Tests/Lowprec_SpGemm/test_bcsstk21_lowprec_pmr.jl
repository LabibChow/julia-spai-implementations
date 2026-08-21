using SparseArrays
using LinearAlgebra
using MatrixDepot
using Plots
using Printf
import Measures


include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "Utils", "dropping.jl"))


spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_pmr_spai.jl"))

include(joinpath(spai_dir, "my_pmr_lowprec_spgemm.jl"))


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


# Experiment settings#

itmax = 1000

tol = 1.0e-15

# Same symmetric dropping used for the PCG experiment.
max_per_column = 100

println("\nExperiment settings:")
println("itmax = $itmax")
println("tol = $tol")
println("max_per_column = $max_per_column")


# Warm-up runs

println("\nCompiling full-precision PMR...")

my_pmr(
    A,
    Pr,
    copy(M0),
    1,
    tol;
    max_per_column = max_per_column
)

println("\nCompiling low-precision-SpGEMM PMR...")

my_pmr_lowprec_spgemm(
    A,
    Pr,
    copy(M0),
    1,
    tol;
    max_per_column = max_per_column
)

GC.gc()


# Full Float64 PMR

println("\n")
println("FULL FLOAT64 PMR")

full64_result = nothing

full64_total_time = @elapsed begin

    full64_result = my_pmr(
        A,
        Pr,
        copy(M0),
        itmax,
        tol;
        max_per_column = max_per_column
    )

end

M_full64, R_full64 = full64_result


# PMR with Float32 A*P and A*M

GC.gc()

println("\n")
println("PMR WITH FLOAT32 A*P AND A*M SpGEMMs")
lowprec_result = nothing

lowprec_total_time = @elapsed begin

    lowprec_result = my_pmr_lowprec_spgemm(
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


# Independent Float64 final residual checks

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
    "%-44s %15.6e\n",
    "Full Float64 final recorded residual:",
    R_full64[end]
)

@printf(
    "%-44s %15.6e\n",
    "Low-precision final recorded residual:",
    R_lowprec[end]
)

@printf(
    "%-44s %15.6e\n",
    "Full Float64 true final residual:",
    true_residual_full64
)

@printf(
    "%-44s %15.6e\n",
    "Low-precision true final residual:",
    true_residual_lowprec
)

@printf(
    "%-44s %15d\n",
    "Full Float64 iterations:",
    length(R_full64) - 1
)

@printf(
    "%-44s %15d\n",
    "Low-precision SpGEMM iterations:",
    length(R_lowprec) - 1
)

@printf(
    "%-44s %15.6f s\n",
    "Full Float64 total time:",
    full64_total_time
)

@printf(
    "%-44s %15.6f s\n",
    "Low-precision total time:",
    lowprec_total_time
)

@printf(
    "%-44s %15.6f\n",
    "Runtime ratio low precision / Float64:",
    lowprec_total_time / full64_total_time
)


# Low-precision SpGEMM timing breakdown

println("\n")
println("LOW-PRECISION SpGEMM TIMING BREAKDOWN")
@printf(
    "%-44s %15.6f s\n",
    "P -> Float32 conversion time:",
    lowprec_timings.P_to_Float32
)

@printf(
    "%-44s %15.6f s\n",
    "Float32 A*P SpGEMM time:",
    lowprec_timings.AP_Float32_SpGEMM
)

@printf(
    "%-44s %15.6f s\n",
    "A*P -> Float64 conversion time:",
    lowprec_timings.AP_to_Float64
)

@printf(
    "%-44s %15.6f s\n",
    "M -> Float32 conversion time:",
    lowprec_timings.M_to_Float32
)

@printf(
    "%-44s %15.6f s\n",
    "Float32 A*M SpGEMM time:",
    lowprec_timings.AM_Float32_SpGEMM
)

@printf(
    "%-44s %15.6f s\n",
    "A*M -> Float64 conversion time:",
    lowprec_timings.AM_to_Float64
)

@printf(
    "%-44s %15.6f s\n",
    "Total conversion time:",
    lowprec_timings.total_conversion
)

@printf(
    "%-44s %15.6f s\n",
    "Total Float32 SpGEMM time:",
    lowprec_timings.total_lowprec_spgemm
)


# Sparsity and symmetry

println("\n")
println("SPARSITY AND SYMMETRY")

@printf(
    "%-44s %15d\n",
    "nnz(M_full64):",
    nnz(M_full64)
)

@printf(
    "%-44s %15d\n",
    "nnz(M_lowprec):",
    nnz(M_lowprec)
)

@printf(
    "%-44s %15.6e\n",
    "nnz(M_full64)/n^2:",
    nnz(M_full64) / n^2
)

@printf(
    "%-44s %15.6e\n",
    "nnz(M_lowprec)/n^2:",
    nnz(M_lowprec) / n^2
)

@printf(
    "%-44s %15.6e\n",
    "Full Float64 symmetry error:",
    symmetry_error_full64
)

@printf(
    "%-44s %15.6e\n",
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
    "%-44s %15.6e\n",
    "Maximum relative-history difference:",
    maximum_history_difference
)


# Convergence plot

plot(
    0:length(relative_R_full64)-1,
    relative_R_full64,
    yscale = :log10,
    linewidth = 2,
    label = "PMR Float64",
    xlabel = "Iteration",
    ylabel = "Relative residual norm",
    title = "bcsstk21: PMR Float64 vs Float32 SpGEMMs",
    legend = :topright,
    grid = true,
    margin = 5Measures.mm
)

plot!(
    0:length(relative_R_lowprec)-1,
    relative_R_lowprec,
    linewidth = 2,
    linestyle = :dash,
    label = "PMR Float32 A*P and A*M"
)


# Save plot

output_path = joinpath(@__DIR__, "bcsstk21_pmr_lowprec_spgemm.png")

savefig(output_path)

println("\nPlot saved to:")
println(output_path)