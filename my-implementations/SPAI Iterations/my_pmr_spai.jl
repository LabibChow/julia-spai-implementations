using SparseArrays
using LinearAlgebra


function my_pmr(A, Pr, M, itmax, tol)
    n = size(A, 1)

    R = spzeros(n, n)
    Z = spzeros(n, n)
    PrA = spzeros(n, n)
    PrAZ = spzeros(n, n)

    R_norm = zeros(itmax + 1)

    PrA .= Pr * A

    R .= I - A * M
    Z .= Pr * R
    PrAZ .= PrA * Z

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        alpha = frob_inner_prod(Z, PrAZ) / frob_norm_squared(PrAZ)

        M .+= alpha .* Z

        R .= I - A * M
        Z .= Pr * R
        PrAZ .= PrA * Z

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_pmr it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M)/n^2)")

        if R_norm[i + 1] < tol
            break
        end
    end

    return M, R_norm[1:i+1]
end