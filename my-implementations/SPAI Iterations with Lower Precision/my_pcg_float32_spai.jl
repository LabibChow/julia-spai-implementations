using SparseArrays
using LinearAlgebra

function my_pcg_float32(A, Pr, M, itmax, tol)
    A64 = Float64.(A)

    A32 = Float32.(A)
    Pr32 = Float32.(Pr)
    M32 = Float32.(M)

    n = size(A32, 1)

    R64 = spzeros(Float64, n, n)

    R32 = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)

    R_norm = zeros(Float64, itmax + 1)

    R64 .= I - A64 * Float64.(M32)

    R32 .= Float32.(R64)
    Z .= Pr32 * R32

    R_Z_frob = frob_inner_prod(R32, Z)

    P .= Z
    AP .= A32 * P

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        P_AP_frob = frob_inner_prod(P, AP)

        alpha = Float32(R_Z_frob / P_AP_frob)

        M32 .+= alpha .* P

        R64 .= I - A64 * Float64.(M32)

        R32 .= Float32.(R64)
        Z .= Pr32 * R32

        old_R_Z_frob = R_Z_frob
        R_Z_frob = frob_inner_prod(R32, Z)

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_pcg_float32 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        beta = Float32(R_Z_frob / old_R_Z_frob)

        P .*= beta
        P .+= Z

        AP .= A32 * P
    end

    return M32, R_norm[1:i+1]
end