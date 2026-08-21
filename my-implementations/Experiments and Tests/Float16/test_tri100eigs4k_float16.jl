using SparseArrays
using LinearAlgebra
using Random
using Printf

utils_dir = joinpath(@__DIR__, "..", "..", "Utils",)

float16_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Float16",)

include(joinpath(utils_dir, "frobenius.jl"))

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

function run_float16_method(
    method_name::String,
    method,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)
    println()
    println("$method_name FLOAT16")
    result = nothing

    construction_time = @elapsed begin
        result = method(
            A,
            Pr16,
            copy(M0_16),
            itmax,
            tol,
        )
    end

    M16, residual_history = result

    final_residual = relative_spai_residual(
        A,
        M16,
    )

    values_finite =
        all(isfinite, nonzeros(M16))

    iterations_completed =
        length(residual_history) - 1

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
        nnz(M16),
    )

    @printf(
        "Density of M:              %.6e\n",
        nnz(M16) / length(M16),
    )

    println(
        "Element type:              ",
        eltype(M16),
    )

    println(
        "All values finite:         ",
        values_finite,
    )

    @printf(
        "Construction time:         %.4f seconds\n",
        construction_time,
    )

    return (
        method = method_name,
        M16 = M16,
        residual_history = residual_history,
        iterations = iterations_completed,
        residual = final_residual,
        nnz_M = nnz(M16),
        density = nnz(M16) / length(M16),
        values_finite = values_finite,
        construction_time = construction_time,
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
println("A element type:  ", eltype(A))
println("Symmetry error:  ", norm(A - transpose(A)))

itmax = 300
tol = 1.0e-15

A_diagonal = diag(A)

if any(iszero, A_diagonal)
    error(
        "The Jacobi preconditioner cannot be formed because " *
        "A contains a zero diagonal entry.",
    )
end

M0_16 = spdiagm(
    0 => ones(Float16, n),
)

Pr16 = spdiagm(
    0 => Float16.(
        1.0 ./ A_diagonal,
    ),
)

println()
println("Experiment parameters")
println("itmax:           ", itmax)
println("tol:             ", tol)
println("A type:          ", eltype(A))
println("Pr16 type:       ", eltype(Pr16))
println("M0_16 type:      ", eltype(M0_16))

println()
println("Testing one PMR iteration for compilation.")

my_pmr_float16(
    A,
    Pr16,
    copy(M0_16),
    1,
    tol,
)

GC.gc()

pmr_float16 = run_float16_method(
    "PMR",
    my_pmr_float16,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)

GC.gc()

println()
println("Testing one PCG iteration for compilation.")

my_pcg_float16(
    A,
    Pr16,
    copy(M0_16),
    1,
    tol,
)

GC.gc()

pcg_float16 = run_float16_method(
    "PCG",
    my_pcg_float16,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)

GC.gc()

println()
println("Testing one LOPMR iteration for compilation.")

my_lopmr_float16(
    A,
    Pr16,
    copy(M0_16),
    1,
    tol,
)

GC.gc()

lopmr_float16 = run_float16_method(
    "LOPMR",
    my_lopmr_float16,
    A,
    Pr16,
    M0_16,
    itmax,
    tol,
)

println()
println("TRI100EIGS4K FLOAT16 SUMMARY")

@printf(
    "%-10s %-14s %-20s %-14s %-14s %-12s\n",
    "Method",
    "Iterations",
    "Relative residual",
    "nnz(M)",
    "Time (s)",
    "Finite",
)

for result in (
    pmr_float16,
    pcg_float16,
    lopmr_float16,
)
    @printf(
        "%-10s %-14d %-20.6e %-14d %-14.4f %-12s\n",
        result.method,
        result.iterations,
        result.residual,
        result.nnz_M,
        result.construction_time,
        string(result.values_finite),
    )
end

println()
println("Float16 experiment completed.")