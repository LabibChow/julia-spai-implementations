using SparseArrays
using LinearAlgebra
using MySPAI: frob_inner_prod, frob_norm, frob_norm_squared

function my_pmr_float32(A, Pr, M, itmax, tol)
    A32 = Float32.(A)
    Pr32 = Float32.(Pr)
    M32 = Float32.(M)

    n = size(A32, 1)

    R = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    PrA = spzeros(Float32, n, n)
    PrAZ = spzeros(Float32, n, n)

    R_norm = zeros(Float32, itmax + 1)

    PrA .= Pr32 * A32
    R .= I - A32 * M32
    Z .= Pr32 * R
    PrAZ .= PrA * Z

    i = 0
    R_norm[i + 1] = frob_norm(R)

    while i < itmax
        alpha = frob_inner_prod(Z, PrAZ) / frob_norm_squared(PrAZ)

        M32 .+= alpha .* Z

        R .= I - A32 * M32
        Z .= Pr32 * R
        PrAZ .= PrA * Z

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println("my_pmr_float32 it = $i, err = $(R_norm[i + 1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if R_norm[i + 1] < tol
            break
        end
    end

    return M32, R_norm[1:i+1]
end