using SparseArrays
using LinearAlgebra

function my_mr_float32(A, M, itmax, tol)
    A64 = Float64.(A)

    A32 = Float32.(A)
    M32 = Float32.(M)

    n = size(A32, 1)

    R64 = spzeros(Float64, n, n)

    R32 = spzeros(Float32, n, n)
    AR = spzeros(Float32, n, n)

    R_norm = zeros(Float64, itmax + 1)

    R64 .= I - A64 * Float64.(M32)

    R32 .= Float32.(R64)
    AR .= A32 * R32

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        alpha = Float32(
            frob_inner_prod(R32, AR) /
            frob_norm_squared(AR)
        )

        M32 .+= alpha .* R32

        R64 .= I - A64 * Float64.(M32)

        R32 .= Float32.(R64)
        AR .= A32 * R32

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_mr_float32 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end
    end

    return M32, R_norm[1:i+1]
end