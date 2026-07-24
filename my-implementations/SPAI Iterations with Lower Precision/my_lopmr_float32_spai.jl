using SparseArrays
using LinearAlgebra


function my_lopmr_float32(A, Pr, M, itmax, tol; max_per_column::Union{Nothing, Int} = nothing,)
    A64 = Float64.(A)

    A32 = Float32.(A)
    Pr32 = Float32.(Pr)
    M32 = Float32.(M)

    n = size(A32, 1)

    R64 = spzeros(Float64, n, n)

    R32 = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AZ = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)
    PrAZ = spzeros(Float32, n, n)
    PrAP = spzeros(Float32, n, n)

    R_norm = zeros(Float64, itmax + 1)

    # Actual residual in Float64.
    R64 .= I - A64 * Float64.(M32)

    R32 .= Float32.(R64)
    Z .= Pr32 * R32

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        AZ .= A32 * Z
        AP .= A32 * P

        PrAZ .= Pr32 * AZ
        PrAP .= Pr32 * AP

        if i == 0
            denominator = frob_inner_prod(AZ, PrAZ)

            if denominator == 0
                @warn "LOPMR stopped because the delta denominator is zero."
                break
            end

            delta = Float32(
                frob_inner_prod(Z, AZ) / denominator
            )

            gamma = Float32(1.0)
        else
            AZ_PrAZ = frob_inner_prod(AZ, PrAZ)
            AP_PrAP = frob_inner_prod(AP, PrAP)
            AZ_PrAP = frob_inner_prod(AZ, PrAP)
            Z_AZ = frob_inner_prod(Z, AZ)
            Z_AP = frob_inner_prod(Z, AP)

            c = AZ_PrAZ * AP_PrAP - AZ_PrAP^2

            if c == 0
                @warn "LOPMR stopped because the 2×2 system is singular."
                break
            end

            delta = Float32(
                (AP_PrAP * Z_AZ - AZ_PrAP * Z_AP) / c
            )

            gamma = Float32(
                (AZ_PrAZ * Z_AP - AZ_PrAP * Z_AZ) / c
            )
        end

        if delta == 0
            @warn "LOPMR stopped because delta is zero."
            break
        end

        # Update the approximate inverse.
        M32 .+= delta .* Z
        M32 .+= gamma .* P

        # Restrict the sparsity immediately after both updates.
        if max_per_column !== nothing
            M32 = drop_largest_per_column(
                M32,
                max_per_column;
                keep_diagonal = true,
            )
        end

        # Recompute the actual residual after dropping.
        R64 .= I - A64 * Float64.(M32)

        R32 .= Float32.(R64)

        P .*= Float32(gamma / delta)
        P .+= Z

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_lopmr_float32 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        Z .= Pr32 * R32
    end

    return M32, R_norm[1:i+1]
end