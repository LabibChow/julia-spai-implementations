using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm

function my_lopmr(A, Pr, M, itmax, tol)
    n = size(A, 1)

    R = spzeros(n, n)
    Z = spzeros(n, n)
    P = spzeros(n, n)
    AZ = spzeros(n, n)
    AP = spzeros(n, n)
    PrAZ = spzeros(n, n)
    PrAP = spzeros(n, n)

    R_norm = zeros(itmax + 1)

    R .= I - A * M
    Z .= Pr * R

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        AZ .= A * Z
        AP .= A * P
        PrAZ .= Pr * AZ
        PrAP .= Pr * AP

        if i == 0
            delta = frob_inner_prod(Z, AZ) / frob_inner_prod(AZ, PrAZ)
            gamma = 1.0
        else
            AZ_PrAZ = frob_inner_prod(AZ, PrAZ)
            AP_PrAP = frob_inner_prod(AP, PrAP)
            AZ_PrAP = frob_inner_prod(AZ, PrAP)
            Z_AZ = frob_inner_prod(Z, AZ)
            Z_AP = frob_inner_prod(Z, AP)

            c = AZ_PrAZ * AP_PrAP - AZ_PrAP^2

            delta = (AP_PrAP * Z_AZ - AZ_PrAP * Z_AP) / c
            gamma = (AZ_PrAZ * Z_AP - AZ_PrAP * Z_AZ) / c
        end

        M .+= delta .* Z
        M .+= gamma .* P

        R .-= delta .* AZ
        R .-= gamma .* AP

        P .*= gamma / delta
        P .+= Z

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_lopmr it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        Z .= Pr * R
    end

    return M, R_norm[1:i+1]
end