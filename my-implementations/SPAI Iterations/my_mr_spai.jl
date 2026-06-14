using SparseArrays
using LinearAlgebra

frob_inner_prod(A, B) = sum(A .* B)
frob_norm_squared(A) = sum(abs2, A)
frob_norm(A) = sqrt(frob_norm_squared(A))

function my_mr(A, M, itmax, tol)
    n = size(A, 1)

    R = spzeros(n, n)
    AR = spzeros(n, n)
    R_norm = zeros(itmax + 1)

    R .= I - A * M
    AR .= A * R

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        alpha = frob_inner_prod(R, AR) / frob_norm_squared(AR)

        M .+= alpha .* R
        R .-= alpha .* AR
        AR .= A * R

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_mr it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M)/n^2)")

        if R_norm[i + 1] < tol
            break
        end
    end

    return M, R_norm[1:i+1]
end