using SparseArrays
using LinearAlgebra


"""
    my_pmr_float32(
        A,
        Pr,
        M,
        itmax,
        tol;
        max_per_column = nothing,
    )

Construct an SPAI using the PMR matrix iteration in Float32.

Precision distribution:
- A, Pr, M, R, Z, PrA, PrAZ and recurrence arithmetic: Float32
- The actual residual used for convergence measurement is recomputed
  in Float64 as

      R64 = I - A64 * Float64(M32)

If `max_per_column` is provided, symmetric dropping is applied
immediately after each update of M and before recomputing the residual.
"""
function my_pmr_float32(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing,
)

    n = size(A, 1)
    # Precision setup

    # Float64 A is used only for accurate residual measurement.
    A64 = SparseMatrixCSC{Float64, Int}(A)

    # Main PMR iteration is Float32.
    A32 = SparseMatrixCSC{Float32, Int}(A)
    Pr32 = SparseMatrixCSC{Float32, Int}(Pr)
    M32 = SparseMatrixCSC{Float32, Int}(M)

    Id64 = spdiagm(
        0 => ones(Float64, n)
    )

    R_norm = zeros(
        Float64,
        itmax + 1
    )

    # Pr*A in Float32

    PrA = Pr32 * A32

    # Initial actual residual in Float64

    R64 = Id64 - A64 * Float64.(M32)


    R32 = Float32.(R64)


    # Initial PMR quantities

    Z = Pr32 * R32

    PrAZ = PrA * Z

    i = 0

    R_norm[1] = frob_norm(R64)

    println(
        "my_pmr_float32 it = 0, ",
        "err = $(R_norm[1]), ",
        "nnz(M)/n^2 = $(nnz(M32) / n^2)"
    )


    # PMR iteration

    while i < itmax

        # Compute alpha

        denominator =
            frob_norm_squared(PrAZ)

        if denominator == 0 || !isfinite(denominator)
    @warn "PMR stopped because the alpha denominator is invalid."
    break
end


        numerator =
            frob_inner_prod(
                Z,
                PrAZ
            )


        alpha =
            Float32(
                numerator /
                denominator
            )


        if !isfinite(alpha)

            @warn(
                "PMR stopped because alpha is not finite."
            )

            break
        end

        # Update M in Float32

        M32 =
            M32 +
            alpha .* Z


        # Optional symmetric dropping

        if max_per_column !== nothing
            M32 =
                drop_largest_per_column(
                    M32,
                    max_per_column;
                    keep_diagonal = true,
                )

        end


        # True residual corresponding to the dropped M, evaluated in Float64 for convergence measurement.

        R64 = Id64 - A64 * Float64.(M32)

        # Back to Float32 for the PMR recurrence

        R32 = Float32.(R64)

        Z = Pr32 * R32

        PrAZ = PrA * Z


        # Residual norm

        i += 1

        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_pmr_float32 it = $i, ",
            "err = $(R_norm[i + 1]), ",
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end
    end

    return M32, R_norm[1:i+1]
end