using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm

function my_lopmr_float32(A, Pr, M, itmax, tol)
    A32 = Float32.(A)
    Pr32 = Float32.(Pr)
    M32 = Float32.(M)

    n = size(A32, 1)

    R = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AZ = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)
    PrAZ = spzeros(Float32, n, n)
    PrAP = spzeros(Float32, n, n)

    R_norm = zeros(Float32, itmax + 1)

    R .= I - A32 * M32
    Z .= Pr32 * R

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        AZ .= A32 * Z
        AP .= A32 * P
        PrAZ .= Pr32 * AZ
        PrAP .= Pr32 * AP

        if i == 0
            delta = frob_inner_prod(Z, AZ) / frob_inner_prod(AZ, PrAZ)
            gamma = Float32(1.0)
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

        M32 .+= delta .* Z
        M32 .+= gamma .* P

        R .-= delta .* AZ
        R .-= gamma .* AP

        P .*= gamma / delta
        P .+= Z

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_lopmr_float32 it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        Z .= Pr32 * R
    end

    return M32, R_norm[1:i+1]
end