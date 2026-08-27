using SparseArrays
using LinearAlgebra

function my_pmr_fully_float32(A, Pr, M, itmax, tol;
    max_per_column::Union{Nothing,Int}=nothing,)

    n = size(A,1)
    A32 = SparseMatrixCSC{Float32,Int}(A)
    Pr32 = SparseMatrixCSC{Float32,Int}(Pr)
    M32 = SparseMatrixCSC{Float32,Int}(M)
    Id32 = spdiagm(0 => ones(Float32,n))
    tol32 = Float32(tol)

    R_norm = zeros(Float32,itmax+1)
    
    R32 = Id32 - A32*M32
    Z = Pr32*R32
    PrA = Pr32*A32
    PrAZ = PrA*Z

    R_norm[1] = frob_norm(R32)

    println("PMR fully F32 it = 0, err = $(R_norm[1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

    i = 0
    while i < itmax
        num = Float32(frob_inner_prod(Z,PrAZ))
        den = Float32(frob_norm_squared(PrAZ))

        if den == 0f0 || !isfinite(den)
            @warn "PMR stopped because alpha denominator is invalid."
            break
        end

        alpha = num/den
        if !isfinite(alpha)
            @warn "PMR stopped because alpha is not finite."
            break
        end

        M32 = M32 + alpha.*Z

        if max_per_column !== nothing
            M32 = drop_largest_per_column(M32,max_per_column; keep_diagonal=true)
        end

        R32 = Id32 - A32*M32
        rnorm32 = frob_norm(R32)

        i += 1
        R_norm[i+1] = rnorm32

        println("PMR fully F32 it = $i, err = $(R_norm[i+1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if rnorm32 < tol32
            break
        end

        Z = Pr32*R32
        PrAZ = PrA*Z
    end

    return M32, R_norm[1:i+1]
end