using SparseArrays
using LinearAlgebra
using Random
using Printf

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations",)

float32_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision",)

float16_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Float16",)

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

include(joinpath(utils_dir, "frobenius.jl"))

include(joinpath(spai_dir, "my_pmr_spai.jl",))

include(joinpath(spai_dir, "my_pcg_spai.jl",))

include(joinpath(spai_dir, "my_lopmr_spai.jl",))

include(joinpath(float32_dir, "my_pmr_float32_spai.jl",))

include(joinpath(
    float32_dir,
    "my_pcg_float32_spai.jl",
))

include(joinpath(float32_dir, "my_lopmr_float32_spai.jl",))

include(joinpath(float16_dir, "my_pmr_float16_spai.jl",))

include(joinpath(float16_dir, "my_pcg_float16_spai.jl",))

include(joinpath(float16_dir, "my_lopmr_float16_spai.jl",))

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

    R64 = I64 - A64 * M64

    return frob_norm(R64) / sqrt(Float64(n))
end

function run_method(
    method_name::String,
    precision_name::String,
    method,
    A,
    Pr,
    M0,
    itmax,
    tol,
)
    println()
    println("$method_name $precision_name")

    result = nothing

    construction_time = @elapsed begin
        result = method(
            A,
            Pr,
            copy(M0),
            itmax,
            tol,
        )
    end

    M_result, residual_history = result

    iterations_completed =
        length(residual_history) - 1

    final_residual =
        relative_spai_residual(
            A,
            M_result,
        )

    finite_values =
        all(isfinite, nonzeros(M_result))

    density =
        nnz(M_result) / length(M_result)

    println()
    println(
        "Iterations completed:      ",
        iterations_completed,
    )

    @printf(
        "Relative Float64 residual: %.6e\n",
        final_residual,
    )

    println(
        "nnz(M):                    ",
        nnz(M_result),
    )

    @printf(
        "Density of M:              %.6e\n",
        density,
    )

    println(
        "Element type:              ",
        eltype(M_result),
    )

    println(
        "All values finite:         ",
        finite_values,
    )

    @printf(
        "Construction time:         %.4f seconds\n",
        construction_time,
    )

    return (
        method = method_name,
        precision = precision_name,
        M = M_result,
        residual_history = residual_history,
        iterations = iterations_completed,
        residual = final_residual,
        nnz_M = nnz(M_result),
        density = density,
        finite = finite_values,
        time = construction_time,
    )
end

Random.seed!(1)

n = 4_000
k = 100

diagonal_entries = vcat(
    0.05 .+ rand(k),
    ones(n - k),
)

subdiagonal_entries =
    rand(n - 1)

B = spdiagm(
     0 => diagonal_entries,
    -1 => subdiagonal_entries,
)

A = sparse(
    B * transpose(B),
)

println("Matrix:          tri100eigs4k")
println("Size:            ", size(A))
println("nnz(A):          ", nnz(A))
println("Density of A:    ", nnz(A) / length(A))
println("Element type:    ", eltype(A))
println("Symmetry error:  ", norm(A - transpose(A)))
itmax = 300
tol = 1.0e-15

println()
println("Experiment parameters")
println("itmax:           ", itmax)
println("tol:             ", tol)

A_diagonal = diag(A)

if any(iszero, A_diagonal)
    error(
        "The Jacobi preconditioner cannot be constructed because " *
        "A contains a zero diagonal entry.",
    )
end

M0_64 = spdiagm(
    0 => ones(Float64, n),
)

Pr64 = spdiagm(
    0 => 1.0 ./ A_diagonal,
)

M0_32 = spdiagm(
    0 => ones(Float32, n),
)

Pr32 = spdiagm(
    0 => Float32.(
        1.0 ./ A_diagonal,
    ),
)

M0_16 = spdiagm(
    0 => ones(Float16, n),
)

Pr16 = spdiagm(
    0 => Float16.(
        1.0 ./ A_diagonal,
    ),
)

pmr_float64 = run_method(
    "PMR",
    "Float64",
    my_pmr,
    A,
    Pr64,
    M0_64,
    itmax,
    tol,
)

GC.gc()

pmr_float32 = run_method(
    "PMR",
    "Float32",
    my_pmr_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
)

GC.gc()

pmr_float16 = run_method(
    "PMR",
    "Float16",
    my_pmr_float16,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)

GC.gc()

pcg_float64 = run_method(
    "PCG",
    "Float64",
    my_pcg,
    A,
    Pr64,
    M0_64,
    itmax,
    tol,
)

GC.gc()

pcg_float32 = run_method(
    "PCG",
    "Float32",
    my_pcg_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
)

GC.gc()

pcg_float16 = run_method(
    "PCG",
    "Float16",
    my_pcg_float16,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)

GC.gc()

lopmr_float64 = run_method(
    "LOPMR",
    "Float64",
    my_lopmr,
    A,
    Pr64,
    M0_64,
    itmax,
    tol,
)

GC.gc()

lopmr_float32 = run_method(
    "LOPMR",
    "Float32",
    my_lopmr_float32,
    A,
    Pr32,
    M0_32,
    itmax,
    tol,
)

GC.gc()

lopmr_float16 = run_method(
    "LOPMR",
    "Float16",
    my_lopmr_float16,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)

all_results = (
    pmr_float64,
    pmr_float32,
    pmr_float16,
    pcg_float64,
    pcg_float32,
    pcg_float16,
    lopmr_float64,
    lopmr_float32,
    lopmr_float16,
)

println()
println("TRI100EIGS4K PRECISION COMPARISON")
println("itmax = $itmax")

@printf(
    "%-10s %-10s %-12s %-20s %-12s %-14s %-10s\n",
    "Method",
    "Precision",
    "Iterations",
    "Relative residual",
    "nnz(M)",
    "Time (s)",
    "Finite",
)

for result in all_results
    @printf(
        "%-10s %-10s %-12d %-20.6e %-12d %-14.4f %-10s\n",
        result.method,
        result.precision,
        result.iterations,
        result.residual,
        result.nnz_M,
        result.time,
        string(result.finite),
    )
end

println()
println("RESIDUAL SUMMARY")

@printf(
    "%-10s %-18s %-18s %-18s\n",
    "Method",
    "Float64",
    "Float32",
    "Float16",
)

for results in (
    (pmr_float64, pmr_float32, pmr_float16),
    (pcg_float64, pcg_float32, pcg_float16),
    (lopmr_float64, lopmr_float32, lopmr_float16),
)
    @printf(
        "%-10s %-18.6e %-18.6e %-18.6e\n",
        results[1].method,
        results[1].residual,
        results[2].residual,
        results[3].residual,
    )
end

println()
println("ITERATION SUMMARY")

@printf(
    "%-10s %-18s %-18s %-18s\n",
    "Method",
    "Float64",
    "Float32",
    "Float16",
)

for results in (
    (pmr_float64, pmr_float32, pmr_float16),
    (pcg_float64, pcg_float32, pcg_float16),
    (lopmr_float64, lopmr_float32, lopmr_float16),
)
    @printf(
        "%-10s %-18d %-18d %-18d\n",
        results[1].method,
        results[1].iterations,
        results[2].iterations,
        results[3].iterations,
    )
end

println()
println("RUNTIME SUMMARY")

@printf(
    "%-10s %-18s %-18s %-18s\n",
    "Method",
    "Float64",
    "Float32",
    "Float16",
)

for results in (
    (pmr_float64, pmr_float32, pmr_float16),
    (pcg_float64, pcg_float32, pcg_float16),
    (lopmr_float64, lopmr_float32, lopmr_float16),
)
    @printf(
        "%-10s %-18.4f %-18.4f %-18.4f\n",
        results[1].method,
        results[1].time,
        results[2].time,
        results[3].time,
    )
end

println()
println("Precision-comparison experiment completed.")