using SparseArrays
using LinearAlgebra


function my_pcg_float32(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing,
)

    n = size(A, 1)

    # Precision setup

    # Float64 copy used only for accurate residual measurement.
    A64 = SparseMatrixCSC{Float64, Int}(A)

    # Main PCG-SPAI iteration is Float32.
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

    # Initial residual
    # Actual residual is measured in Float64.

    R64 = Id64 - A64 * Float64.(M32)

    # Convert residual back to Float32 for the iteration.
    R32 = Float32.(R64)


    # Initial PCG quantities

    Z = Pr32 * R32

    R_Z_frob = frob_inner_prod(R32, Z)

    P = copy(Z)

    AP = A32 * P

    i = 0

    R_norm[1] =
        frob_norm(R64)

    println(
        "my_pcg_float32 it = 0, ",
        "err = $(R_norm[1]), ",
        "nnz(M)/n^2 = $(nnz(M32) / n^2)"
    )

    # PCG iteration

    while i < itmax

        # Compute alpha

        P_AP_frob =
            frob_inner_prod(
                P,
                AP
            )

        if P_AP_frob == 0 || !isfinite(P_AP_frob)
    @warn "PCG stopped because the alpha denominator is invalid."
    break
end

        alpha =
            Float32(
                R_Z_frob /
                P_AP_frob
            )


        if !isfinite(alpha)

            @warn(
                "PCG stopped because alpha is not finite."
            )

            break
        end

        # Update M in Float32

        M32 = M32 + alpha .* P

        # Optional symmetric dropping

        if max_per_column !== nothing
         M32 =
                drop_largest_per_column(
                    M32,
                    max_per_column;
                    keep_diagonal = true,
                )

        end


        # Actual residual in Float64
        R64 = Id64 - A64 * Float64.(M32)

        # Float32 residual for the PCG recurrence
        R32 = Float32.(R64)

        # Preconditioned residual in Float32

        Z =
            Pr32 * R32


        old_R_Z_frob =
            R_Z_frob

        R_Z_frob =
            frob_inner_prod(
                R32,
                Z
            )


        # Residual norm

        i += 1

        R_norm[i + 1] =
            frob_norm(R64)

        println(
            "my_pcg_float32 it = $i, ",
            "err = $(R_norm[i + 1]), ",
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )


        if R_norm[i + 1] < tol
            break
        end


        # Compute beta

        if old_R_Z_frob == 0 || !isfinite(old_R_Z_frob)

    @warn(
        "PCG stopped because previous ",
        "(R, Z)_F is invalid."
    )

    break
end


        beta =
            Float32(
                R_Z_frob /
                old_R_Z_frob
            )


        if !isfinite(beta)

            @warn(
                "PCG stopped because beta is not finite."
            )

            break
        end


        # Search-direction update in Float32
        P = Z + beta .* P

        # A*P in Float32
           AP = A32 * P
    end


    return M32, R_norm[1:i+1]
end