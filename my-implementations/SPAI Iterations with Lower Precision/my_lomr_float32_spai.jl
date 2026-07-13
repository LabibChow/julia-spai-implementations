using SparseArrays
using LinearAlgebra

function my_lomr_float32(A, M, itmax, tol)
    A64 = Float64.(A)

    A32 = Float32.(A)
    M32 = Float32.(M)

    n = size(A32, 1)

    R64 = spzeros(Float64, n, n)

    R32 = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AZ = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)

    R_norm = zeros(Float64, itmax + 1)

    R64 .= I - A64 * Float64.(M32)

    R32 .= Float32.(R64)
    Z .= R32

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        AZ .= A32 * Z
        AP .= A32 * P

        if i == 0
            delta = Float32(
                frob_inner_prod(Z, AZ) /
                frob_inner_prod(AZ, AZ)
            )

            gamma = Float32(1.0)
        else
            AZ_AZ = frob_inner_prod(AZ, AZ)
            AP_AP = frob_inner_prod(AP, AP)
            AZ_AP = frob_inner_prod(AZ, AP)
            Z_AZ = frob_inner_prod(Z, AZ)
            Z_AP = frob_inner_prod(Z, AP)

            c = AZ_AZ * AP_AP - AZ_AP^2

            delta = Float32(
                (AP_AP * Z_AZ - AZ_AP * Z_AP) / c
            )

            gamma = Float32(
                (AZ_AZ * Z_AP - AZ_AP * Z_AZ) / c
            )
        end

        M32 .+= delta .* Z
        M32 .+= gamma .* P

        R64 .= I - A64 * Float64.(M32)

        R32 .= Float32.(R64)

        P .*= Float32(gamma / delta)
        P .+= Z

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_lomr_float32 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        Z .= R32
    end

    return M32, R_norm[1:i+1]
end