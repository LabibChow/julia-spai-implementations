using SparseArrays
using LinearAlgebra


function my_lomr(A, M, itmax, tol)
    n = size(A, 1)

    R = spzeros(n, n)
    Z = spzeros(n, n)
    P = spzeros(n, n)
    AZ = spzeros(n, n)
    AP = spzeros(n, n)

    R_norm = zeros(itmax + 1)

    R .= I - A * M
    Z .= R

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        AZ .= A * Z
        AP .= A * P

        if i == 0
            delta = frob_inner_prod(Z, AZ) / frob_inner_prod(AZ, AZ)
            gamma = 1.0
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

        M .+= delta .* Z
        M .+= gamma .* P

        R .= I - A * M

        P .*= gamma / delta
        P .+= Z

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_lomr it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        Z .= R
    end

    return M, R_norm[1:i+1]
end