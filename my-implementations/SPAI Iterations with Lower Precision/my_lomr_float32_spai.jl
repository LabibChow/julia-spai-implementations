using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm

function my_lomr_float32(A, M, itmax, tol)
    A32 = Float32.(A)
    M32 = Float32.(M)

    n = size(A32, 1)

    R = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AZ = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)

    R_norm = zeros(Float32, itmax + 1)

    R .= I - A32 * M32
    Z .= R

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        AZ .= A32 * Z
        AP .= A32 * P

        if i == 0
            delta = frob_inner_prod(Z, AZ) / frob_inner_prod(AZ, AZ)
            gamma = Float32(1.0)
        else
            AZ_AZ = frob_inner_prod(AZ, AZ)
            AP_AP = frob_inner_prod(AP, AP)
            AZ_AP = frob_inner_prod(AZ, AP)
            Z_AZ = frob_inner_prod(Z, AZ)
            Z_AP = frob_inner_prod(Z, AP)

            c = AZ_AZ * AP_AP - AZ_AP^2

            delta = (AP_AP * Z_AZ - AZ_AP * Z_AP) / c
            gamma = (AZ_AZ * Z_AP - AZ_AP * Z_AZ) / c
        end

        M32 .+= delta .* Z
        M32 .+= gamma .* P

        R .-= delta .* AZ
        R .-= gamma .* AP

        P .*= gamma / delta
        P .+= Z

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_lomr_float32 it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        Z .= R
    end

    return M32, R_norm[1:i+1]
end