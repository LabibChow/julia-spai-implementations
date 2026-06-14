using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm, frob_norm_squared

function my_mr_float32(A, M, itmax, tol)
    A32 = Float32.(A)
    M32 = Float32.(M)

    n = size(A32, 1)

    R = spzeros(Float32, n, n)
    AR = spzeros(Float32, n, n)
    R_norm = zeros(Float32, itmax + 1)

    R .= I - A32 * M32
    AR .= A32 * R

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        alpha = frob_inner_prod(R, AR) / frob_norm_squared(AR)

        M32 .+= alpha .* R
        R .-= alpha .* AR
        AR .= A32 * R

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_mr_float32 it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if R_norm[i + 1] < tol
            break
        end
    end

    return M32, R_norm[1:i+1]
end