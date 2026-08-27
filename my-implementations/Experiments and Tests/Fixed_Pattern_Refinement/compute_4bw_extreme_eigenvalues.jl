using MatrixMarket
using SparseArrays
using LinearAlgebra
using Serialization
using Arpack
using Printf

# SETTINGS

run_tags = ["it5_drop100", "it20_drop100", "it100_drop100",]

# EXTREME EIGENVALUES OF SYMMETRIZED SPAI
function extreme_eigenvalues(M)

    println("Converting to Float64...")

    M64 = sparse(Float64.(M))

    println("Symmetrizing...")

    Msym = sparse(
        (M64 + transpose(M64)) / 2
    )

    M64 = nothing
    GC.gc()


    println("Computing minimum eigenvalue...")

    λmin = real(
            Arpack.eigs(
                Msym;
                nev = 1,
                ncv = 100,
                which = :SR,
                ritzvec = false,
                tol = 1.0e-8,
                maxiter = 50000,
            )[1][1]
        )

    println("Computing maximum eigenvalue...")

    λmax =
        real(
            Arpack.eigs(
                Msym;
                nev = 1,
                ncv = 100,
                which = :LR,
                ritzvec = false,
                tol = 1.0e-8,
                maxiter = 50000,
            )[1][1]
        )

    Msym = nothing
    GC.gc()

    return λmin, λmax
end


# PROCESS ONE SAVED SPAI

function process_saved_matrix(
    method_name,
    precision_name,
    filename,
    matrix_dir,
)

    path = joinpath(
        matrix_dir,
        filename,
    )

    if !isfile(path)
        error("Saved matrix not found:\n$path")
    end


    println()
    println("$method_name — $precision_name")

    println("Loading:")
    println(path)

    M = deserialize(path)

    println("Size:   ", size(M))
    println("nnz(M): ", nnz(M))


    λmin, λmax =
        extreme_eigenvalues(M)

    println()

    @printf(
        "lambda_min = %.16e\n",
        λmin,
    )

    @printf(
        "lambda_max = %.16e\n",
        λmax,
    )

    M = nothing
    GC.gc()


    return (
        method = method_name,
        precision = precision_name,
        λmin = λmin,
        λmax = λmax,
    )
end

# PROCESS ONE ENDPOINT

function process_endpoint(
    run_tag,
    λmin_A,
    λmax_A,
    λmin_Ainv,
    λmax_Ainv,
)

    println()
    println("PROCESSING ENDPOINT: $run_tag")


    matrix_dir = joinpath(@__DIR__, "4bw100eigs20k_saved_matrices", run_tag,)

    results_dir = joinpath(@__DIR__, "4bw100eigs20k_precision_results", run_tag,)

    mkpath(results_dir)

    output_path = joinpath(
        results_dir,
        "extreme_eigenvalues.txt",
    )

    results = []

    # PMR
    push!(
        results,
        process_saved_matrix(
            "PMR",
            "Float64",
            "pmr_float64_M.jls",
            matrix_dir,
        ),
    )

    push!(
        results,
        process_saved_matrix(
            "PMR",
            "Float32",
            "pmr_float32_M.jls",
            matrix_dir,
        ),
    )

    push!(
        results,
        process_saved_matrix(
            "PMR",
            "Float32 refined",
            "pmr_float32_refined_M.jls",
            matrix_dir,
        ),
    )

    # PCG
    push!(
        results,
        process_saved_matrix(
            "PCG",
            "Float64",
            "pcg_float64_M.jls",
            matrix_dir,
        ),
    )

    push!(
        results,
        process_saved_matrix(
            "PCG",
            "Float32",
            "pcg_float32_M.jls",
            matrix_dir,
        ),
    )

    push!(
        results,
        process_saved_matrix(
            "PCG",
            "Float32 refined",
            "pcg_float32_refined_M.jls",
            matrix_dir,
        ),
    )

    # LOPMR
    push!(
        results,
        process_saved_matrix(
            "LOPMR",
            "Float64",
            "lopmr_float64_M.jls",
            matrix_dir,
        ),
    )

    push!(
        results,
        process_saved_matrix(
            "LOPMR",
            "Float32",
            "lopmr_float32_M.jls",
            matrix_dir,
        ),
    )

    push!(
        results,
        process_saved_matrix(
            "LOPMR",
            "Float32 refined",
            "lopmr_float32_refined_M.jls",
            matrix_dir,
        ),
    )

    # SAVE
    open(output_path, "w") do io

        println(io, "Matrix: 4bw100eigs20k")
        println(io, "Endpoint: $run_tag")
        println(io)

        @printf(
            io,
            "lambda_min(A)    = %.16e\n",
            λmin_A,
        )

        @printf(
            io,
            "lambda_max(A)    = %.16e\n",
            λmax_A,
        )

        @printf(
            io,
            "lambda_min(A^-1) = %.16e\n",
            λmin_Ainv,
        )

        @printf(
            io,
            "lambda_max(A^-1) = %.16e\n",
            λmax_Ainv,
        )

        println(io)

        @printf(
            io,
            "%-10s %-20s %-24s %-24s\n",
            "Method",
            "Precision",
            "lambda_min(M)",
            "lambda_max(M)",
        )


        for result in results

            @printf(
                io,
                "%-10s %-20s %-24.16e %-24.16e\n",
                result.method,
                result.precision,
                result.λmin,
                result.λmax,
            )

        end
    end

    # SCREEN SUMMARY
    println()
    println("EXTREME EIGENVALUE SUMMARY — $run_tag")

    @printf(
        "%-10s %-20s %-18s %-18s\n",
        "Method",
        "Precision",
        "lambda_min",
        "lambda_max",
    )

    @printf(
        "%-10s %-20s %-18.6e %-18.6e\n",
        "A^-1",
        "Reference",
        λmin_Ainv,
        λmax_Ainv,
    )

    for result in results

        @printf(
            "%-10s %-20s %-18.6e %-18.6e\n",
            result.method,
            result.precision,
            result.λmin,
            result.λmax,
        )

    end

    println()
    println("Results saved to:")
    println(output_path)

    return nothing
end

# MAIN
function main()

    println()
    println("4bw100eigs20k — EXTREME EIGENVALUES")
    println("Endpoints: 5, 20, 100")

    # COMPUTE A / A^-1 REFERENCE ONCE

    matrix_path = joinpath(@__DIR__, "4bw100eigs20k.mtx",)

    if !isfile(matrix_path)
        error("Matrix file not found:\n$matrix_path")
    end


    println()
    println("REFERENCE MATRIX A")


    A_raw =
        MatrixMarket.mmread(
            matrix_path
        )

    A =
        sparse(
            Float64.(
                sparse(A_raw)
            )
        )

    A_raw = nothing
    GC.gc()

    println("Size:   ", size(A))
    println("nnz(A): ", nnz(A))

    println()
    println("Computing lambda_max(A)...")

    λmax_A =
        real(
            Arpack.eigs(
                A;
                nev = 1,
                which = :LR,
                ritzvec = false,
                tol = 1.0e-10,
                maxiter = 10000,
            )[1][1]
        )


    println("Computing lambda_min(A)...")

    λmin_A =
        real(
            Arpack.eigs(
                A;
                nev = 1,
                sigma = 0.0,
                which = :LM,
                ritzvec = false,
                tol = 1.0e-10,
                maxiter = 10000,
            )[1][1]
        )

    λmin_Ainv = 1.0 / λmax_A
    λmax_Ainv = 1.0 / λmin_A

    println()

    @printf(
        "lambda_min(A)      = %.16e\n",
        λmin_A,
    )

    @printf(
        "lambda_max(A)      = %.16e\n",
        λmax_A,
    )

    @printf(
        "lambda_min(A^-1)   = %.16e\n",
        λmin_Ainv,
    )

    @printf(
        "lambda_max(A^-1)   = %.16e\n",
        λmax_Ainv,
    )

    A = nothing
    GC.gc()

    # PROCESS 5, 20, 100

    for run_tag in run_tags

        process_endpoint(
            run_tag,
            λmin_A,
            λmax_A,
            λmin_Ainv,
            λmax_Ainv,
        )

        GC.gc()
    end


    println()
    println("ALL EXTREME EIGENVALUE ANALYSES COMPLETE")
    println("Endpoints: 5, 20, 100")

end

main()