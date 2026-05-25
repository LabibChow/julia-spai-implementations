using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm

function my_pcg(A, Pr, M, itmax, tol)
    n = size(A, 1)

    R = spzeros(n, n)
    Z = spzeros(n, n)
    P = spzeros(n, n)
    AP = spzeros(n, n)

    R_norm = zeros(itmax + 1)

    R .= I - A * M
    Z .= Pr * R

    R_Z_frob = frob_inner_prod(R, Z)

    P .= Z
    AP .= A * P

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        P_AP_frob = frob_inner_prod(P, AP)

        alpha = R_Z_frob / P_AP_frob

        M .+= alpha .* P
        R .-= alpha .* AP

        Z .= Pr * R

        old_R_Z_frob = R_Z_frob
        R_Z_frob = frob_inner_prod(R, Z)

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_pcg it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        beta = R_Z_frob / old_R_Z_frob

        P .*= beta
        P .+= Z

        AP .= A * P
    end

    return M, R_norm[1:i+1]
end