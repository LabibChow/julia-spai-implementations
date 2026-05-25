using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm_squared

function my_cg_float32(A, M, itmax, tol)
    A32 = Float32.(A)
    M32 = Float32.(M)

    n = size(A32, 1)

    R = spzeros(Float32, n, n)
    P = spzeros(Float32, n, n)
    AP = spzeros(Float32, n, n)

    R_norm = zeros(Float32, itmax + 1)

    R .= I - A32 * M32
    R_frob_norm_squared = frob_norm_squared(R)

    P .= R
    AP .= A32 * P

    i = 0
    R_norm[i + 1] = sqrt(R_frob_norm_squared)

    while i < itmax
        P_AP_frob = frob_inner_prod(P, AP)
        alpha = R_frob_norm_squared / P_AP_frob

        M32 .+= alpha .* P
        R .-= alpha .* AP

        old_R_frob_norm_squared = R_frob_norm_squared
        R_frob_norm_squared = frob_norm_squared(R)

        i += 1
        R_norm[i + 1] = sqrt(R_frob_norm_squared)

        println("my_cg_float32 it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        beta = R_frob_norm_squared / old_R_frob_norm_squared

        P .*= beta
        P .+= R

        AP .= A32 * P
    end

    return M32, R_norm[1:i+1]
end