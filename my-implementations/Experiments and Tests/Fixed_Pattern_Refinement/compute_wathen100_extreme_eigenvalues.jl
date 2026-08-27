using MatrixDepot
using SparseArrays
using LinearAlgebra
using Serialization
using Arpack
using Printf
using DelimitedFiles

# SETTINGS
iteration_endpoints = (5, 20, 100)
max_per_column = 900

results_root = joinpath(@__DIR__, "wathen100_precision_results",)

matrix_root = joinpath(@__DIR__, "wathen100_saved_matrices",)

mkpath(results_root)

output_txt = joinpath( results_root, "extreme_eigenvalues_5_20_100.txt",)

output_csv = joinpath(results_root, "extreme_eigenvalues_5_20_100.csv",)


# EXTREME EIGENVALUES OF A SYMMETRIC SPAI
# The saved matrix is converted to Float64 and symmetrized:
#     M_sym = (M + M') / 2
# I have only computed lambda_min and lambda_max
function extreme_eigenvalues(
    M::SparseMatrixCSC,
)

    M64 = sparse(
            Float64.(M)
    )


    Msym = sparse(
            (M64 + transpose(M64)) / 2
    )

    M64 = nothing
    GC.gc()

    println("  computing lambda_min...")

    λmin = real(
            Arpack.eigs(
                Msym;
                nev = 1,
                which = :SR,
                ritzvec = false,
                tol = 1.0e-8,
                maxiter = 20000,
            )[1][1]
        )

    println("  computing lambda_max...")

    λmax = real(
            Arpack.eigs(
                Msym;
                nev = 1,
                which = :LR,
                ritzvec = false,
                tol = 1.0e-8,
                maxiter = 20000,
            )[1][1]
        )

    Msym = nothing
    GC.gc()

    return λmin, λmax
end

# PROCESS ONE SAVED MATRIX
function process_saved_matrix(
    endpoint,
    method_name,
    precision_name,
    filename,
)

    run_tag =
        "it$(endpoint)_drop$(max_per_column)"


    path = joinpath(matrix_root, run_tag, filename,)

    if !isfile(path)

        println()
        println("SKIPPING missing matrix:")

        println(path)

        return nothing
    end


    println()
    println("$method_name — $precision_name — endpoint $endpoint")

    println("Loading: ", path,)

    M = deserialize(path)


    println("  size   = ", size(M),)
    println("  nnz(M) = ", nnz(M),)

    λmin,
    λmax =
        extreme_eigenvalues(
            M
        )


    @printf(
        "  lambda_min = %.16e\n",
        λmin,
    )

    @printf(
        "  lambda_max = %.16e\n",
        λmax,
    )

    M = nothing
    GC.gc()

    return (
        endpoint =
            endpoint,

        method =
            method_name,

        precision =
            precision_name,

        λmin =
            λmin,

        λmax =
            λmax,
    )
end

# MAIN
function main()

    println()
    println("wathen100 — EXTREME EIGENVALUES AT 5 / 20 / 100")

    # LOAD THE SUITESPARSE MATRIX
    matrix_name = "GHS_psdef/wathen100"


    println()
    println("Loading reference matrix: ", matrix_name,)


    md = mdopen(
            matrix_name,
        )


    A = sparse(
            Float64.(
                md.A
            )
        )


    md = nothing
    GC.gc()


    println(
        "size(A) = ",
        size(A),
    )

    println(
        "nnz(A)  = ",
        nnz(A),
    )

    # REFERENCE EXTREME EIGENVALUES
    println()
    println( "Computing lambda_max(A)...")

    λmax_A =
        real(
            Arpack.eigs(
                A;
                nev = 1,
                which = :LR,
                ritzvec = false,
                tol = 1.0e-10,
                maxiter = 20000,
            )[1][1]
        )

    println(
        "Computing lambda_min(A)..."
    )

    # Shift-invert around zero targets the smallest eigenvalue of this SPD matrix.
    λmin_A =
        real(
            Arpack.eigs(
                A;
                nev = 1,
                sigma = 0.0,
                which = :LM,
                ritzvec = false,
                tol = 1.0e-10,
                maxiter = 20000,
            )[1][1]
        )

    λmin_Ainv =
        1.0 /
        λmax_A

    λmax_Ainv =
        1.0 /
        λmin_A

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

    # SPAI MATRICES
    rows = NamedTuple[]

    specifications = (
        ("PMR",   "Float64",         "pmr_float64_M.jls"),
        ("PMR",   "Float32",         "pmr_float32_M.jls"),
        ("PMR",   "Float32 refined", "pmr_float32_refined_M.jls"),
        ("PMR",   "Float16",         "pmr_float16_M.jls"),
        ("PMR",   "Float16 refined", "pmr_float16_refined_M.jls"),

        ("PCG",   "Float64",         "pcg_float64_M.jls"),
        ("PCG",   "Float32",         "pcg_float32_M.jls"),
        ("PCG",   "Float32 refined", "pcg_float32_refined_M.jls"),
        ("PCG",   "Float16",         "pcg_float16_M.jls"),
        ("PCG",   "Float16 refined", "pcg_float16_refined_M.jls"),

        ("LOPMR", "Float64",         "lopmr_float64_M.jls"),
        ("LOPMR", "Float32",         "lopmr_float32_M.jls"),
        ("LOPMR", "Float32 refined", "lopmr_float32_refined_M.jls"),
        ("LOPMR", "Float16",         "lopmr_float16_M.jls"),
        ("LOPMR", "Float16 refined", "lopmr_float16_refined_M.jls"),
    )

    for endpoint in iteration_endpoints

        for (
            method_name,
            precision_name,
            filename,
        ) in specifications

            result =
                process_saved_matrix(
                    endpoint,
                    method_name,
                    precision_name,
                    filename,
                )


            if result !== nothing

                push!(
                    rows,
                    result,
                )

            end
        end
    end

    # SAVE TEXT OUTPUT
    open(
        output_txt,
        "w",
    ) do io

        println(
            io,
            "Matrix: wathen100",
        )

        println(
            io,
            "SuiteSparse name: GHS_psdef/wathen100",
        )

        println(
            io,
            "Endpoints: 5, 20, 100",
        )

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

        for endpoint in iteration_endpoints

            println(io)

            println(
                io,
                "Endpoint: ",
                endpoint,
            )


            @printf(
                io,
                "%-10s %-20s %-24s %-24s\n",
                "Method",
                "Precision",
                "lambda_min(M)",
                "lambda_max(M)",
            )

            for result in rows

                if result.endpoint ==
                   endpoint

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
        end
    end

    # CSV OUTPUT
    open(
        output_csv,
        "w",
    ) do io

        println(
            io,
            "endpoint,method,precision,lambda_min_M,lambda_max_M"
        )

        for result in rows

            println(
                io,
                join(
                    [
                        string(result.endpoint),
                        result.method,
                        result.precision,
                        string(result.λmin),
                        string(result.λmax),
                    ],
                    ",",
                ),
            )

        end
    end

    # SCREEN SUMMARY
    println()
    println("FINAL EXTREME EIGENVALUE SUMMARY")

    @printf(
        "%-6s %-10s %-20s %-18s %-18s\n",
        "Iter",
        "Method",
        "Precision",
        "lambda_min",
        "lambda_max",
    )

    for result in rows

        @printf(
            "%-6d %-10s %-20s %-18.6e %-18.6e\n",
            result.endpoint,
            result.method,
            result.precision,
            result.λmin,
            result.λmax,
        )

    end

    println()
    println(
        "Reference A^-1:"
    )

    @printf(
        "lambda_min(A^-1) = %.16e\n",
        λmin_Ainv,
    )

    @printf(
        "lambda_max(A^-1) = %.16e\n",
        λmax_Ainv,
    )

    println()
    println(
        "Text output: ",
        output_txt,
    )

    println(
        "CSV output:  ",
        output_csv,
    )

end

main()