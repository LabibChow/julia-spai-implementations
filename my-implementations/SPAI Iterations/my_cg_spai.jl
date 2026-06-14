using SparseArrays
using LinearAlgebra

function frob_inner_prod(A, B)
    return sum(A .* B)
end

function frob_norm_squared(A)
    return frob_inner_prod(A, A)
end

function my_cg(A, M, itmax, tol)
    #A  = original SPD matrix
    #M  = approximate inverse
    #R  = I - A*M
    #P  = search direction
    #AP = A*P

    n = size(A, 1)

    R = spzeros(n, n)
    P = spzeros(n, n)
    AP = spzeros(n, n)

    R_norm = zeros(itmax + 1)

    R .= I - A * M
    R_frob_norm_squared = frob_norm_squared(R)

    P .= R
    AP .= A * P

    i = 0
    R_norm[i + 1] = sqrt(R_frob_norm_squared)

    while i < itmax
        P_AP_frob = frob_inner_prod(P, AP)

        alpha = R_frob_norm_squared / P_AP_frob

        M .+= alpha .* P
        R .-= alpha .* AP

        old_R_frob_norm_squared = R_frob_norm_squared
        R_frob_norm_squared = frob_norm_squared(R)

        i += 1
        R_norm[i + 1] = sqrt(R_frob_norm_squared)

        println("my_cg it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M)/n^2)")

        if R_norm[i + 1] < tol
            break
        end

        beta = R_frob_norm_squared / old_R_frob_norm_squared

        P .*= beta
        P .+= R

        AP .= A * P
    end

    return M, R_norm[1:i+1]
end