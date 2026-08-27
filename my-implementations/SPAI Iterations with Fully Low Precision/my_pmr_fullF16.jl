using SparseArrays
using LinearAlgebra

function my_pmr_fully_float16(A, Pr, M, itmax, tol;
    max_per_column::Union{Nothing,Int}=nothing,)

    n = size(A,1)
    A16 = SparseMatrixCSC{Float16,Int}(A)
    Pr16 = SparseMatrixCSC{Float16,Int}(Pr)
    M16 = SparseMatrixCSC{Float16,Int}(M)
    Id16 = spdiagm(0 => ones(Float16,n))
    tol16 = Float16(tol)

    R_norm = zeros(Float16,itmax+1)

    R16 = Id16 - A16*M16
    Z = Pr16*R16
    PrA = Pr16*A16
    PrAZ = PrA*Z

    R_norm[1] = frob_norm(R16)

    println("PMR fully F16 it = 0, err = $(R_norm[1]), nnz(M)/n^2 = $(nnz(M16)/n^2)")

    i = 0
    while i < itmax
        num = Float16(frob_inner_prod(Z,PrAZ))
        den = Float16(frob_norm_squared(PrAZ))

        if den == Float16(0) || !isfinite(den)
            @warn "PMR stopped because alpha denominator is invalid."
            break
        end

        alpha = num/den
        if !isfinite(alpha)
            @warn "PMR stopped because alpha is not finite."
            break
        end

        M16 = M16 + alpha.*Z

        if max_per_column !== nothing
            M16 = drop_largest_per_column(M16,max_per_column; keep_diagonal=true)
        end

        R16 = Id16 - A16*M16
        rnorm16 = frob_norm(R16)

        i += 1
        R_norm[i+1] = rnorm16

        println("PMR fully F16 it = $i, err = $(R_norm[i+1]), nnz(M)/n^2 = $(nnz(M16)/n^2)")

        if rnorm16 < tol16
            break
        end

        Z = Pr16*R16
        PrAZ = PrA*Z
    end

    return M16, R_norm[1:i+1]
end