using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm

function my_pcg_float32(A, Pr, M, itmax, tol)
    A32 = Float32.(A)
    Pr32 = Float32.(Pr)
    M32 = Float32.(M)

    n = size(A32, 1)

    R = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)

    R_norm = zeros(Float32, itmax + 1)

    R .= I - A32 * M32
    Z .= Pr32 * R

    R_Z_frob = frob_inner_prod(R, Z)

    P .= Z
    AP .= A32 * P

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        P_AP_frob = frob_inner_prod(P, AP)

        alpha = R_Z_frob / P_AP_frob

        M32 .+= alpha .* P
        R .-= alpha .* AP

        Z .= Pr32 * R

        old_R_Z_frob = R_Z_frob
        R_Z_frob = frob_inner_prod(R, Z)

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_pcg_float32 it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        beta = R_Z_frob / old_R_Z_frob

        P .*= beta
        P .+= Z

        AP .= A32 * P
    end

    return M32, R_norm[1:i+1]
end