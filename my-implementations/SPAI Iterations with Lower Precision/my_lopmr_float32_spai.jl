using SparseArrays
using LinearAlgebra


function my_lopmr_float32(
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

    # Main LOPMR iteration is Float32.
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


    # Initial actual residual in Float64

    R64 = Id64 - A64 * Float64.(M32)


    # Float32 residual used by LOPMR.
    R32 = Float32.(R64)


    # Initial LOPMR quantities

    Z = Pr32 * R32

    P = spzeros(Float32, n, n)


    i = 0

    R_norm[1] = frob_norm(R64)


    println(
        "my_lopmr_float32 it = 0, ",
        "err = $(R_norm[1]), ",
        "nnz(M)/n^2 = $(nnz(M32) / n^2)"
    )


    # LOPMR iteration

    while i < itmax

        # Float32 matrix products
        AZ = A32 * Z

        PrAZ = Pr32 * AZ

        # First iteration

        if i == 0

            denominator =
                frob_inner_prod(
                    AZ,
                    PrAZ
                )


            if !isfinite(denominator) ||
               abs(denominator) <= eps(Float32)

                @warn(
                    "LOPMR stopped because the initial ",
                    "delta denominator is invalid or nearly zero."
                )

                break
            end


            delta =
                Float32(
                    frob_inner_prod(
                        Z,
                        AZ
                    ) /
                    denominator
                )


            gamma =
                Float32(1.0)


        # Remaining iterations

        else

            AP = A32 * P

            PrAP = Pr32 * AP

            AZ_PrAZ = frob_inner_prod(AZ, PrAZ)

            AP_PrAP = frob_inner_prod(AP, PrAP)

            AZ_PrAP = frob_inner_prod(AZ, PrAP)

            Z_AZ = frob_inner_prod( Z, AZ)

            Z_AP = frob_inner_prod(Z, AP)


            c = AZ_PrAZ * AP_PrAP - AZ_PrAP^2


            if c == 0 || !isfinite(c)
    @warn "LOPMR stopped because the 2×2 system is invalid."
    break
end


            delta =
                Float32(
                    (
                        AP_PrAP * Z_AZ -
                        AZ_PrAP * Z_AP
                    ) / c
                )


            gamma =
                Float32(
                    (
                        AZ_PrAZ * Z_AP -
                        AZ_PrAP * Z_AZ
                    ) / c
                )
        end


        # Coefficient checks

        if !isfinite(delta) ||
           !isfinite(gamma)

            @warn(
                "LOPMR stopped because delta or gamma ",
                "is not finite."
            )

            break
        end


        if abs(delta) <= eps(Float32)

            @warn(
                "LOPMR stopped because delta is nearly zero."
            )

            break
        end


        # Complete M update in Float32

        M32 = M32 + delta .* Z + gamma .* P


        # Optional symmetric dropping

        if max_per_column !== nothing
            M32 =
                drop_largest_per_column(
                    M32,
                    max_per_column;
                    keep_diagonal = true,
                )

        end

        # True residual corresponding to dropped M, evaluated in Float64.
        R64 = Id64 - A64 * Float64.(M32)

        # Float32 residual for the recurrence
        R32 = Float32.(R64)

        # Search-direction update
        # Uses the current Z, matching the original LOPMR recurrence.

        P =
            Z +
            Float32(gamma / delta) .* P

        # Residual norm

        i += 1

        R_norm[i + 1] = frob_norm(R64)


        println(
            "my_lopmr_float32 it = $i, ",
            "err = $(R_norm[i + 1]), ",
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )


        if R_norm[i + 1] < tol
            break
        end

        # Preconditioned residual for next iteration
        Z = Pr32 * R32

    end

    return M32, R_norm[1:i+1]
end