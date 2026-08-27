using SparseArrays
using LinearAlgebra

function my_pcg_fully_float32(A, Pr, M, itmax, tol;
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
    RZ = Float32(frob_inner_prod(R32,Z))
    P = copy(Z)
    AP = A32*P

    R_norm[1] = frob_norm(R32)

    println("PCG fully F32 it = 0, err = $(R_norm[1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

    i = 0
    while i < itmax
        PAP = Float32(frob_inner_prod(P,AP))

        if PAP == 0f0 || !isfinite(PAP)
            @warn "PCG stopped because alpha denominator is invalid."
            break
        end

        alpha = RZ/PAP
        if !isfinite(alpha)
            @warn "PCG stopped because alpha is not finite."
            break
        end

        M32 = M32 + alpha.*P

        if max_per_column !== nothing
            M32 = drop_largest_per_column(M32,max_per_column; keep_diagonal=true)
        end

        R32 = Id32 - A32*M32
        rnorm32 = frob_norm(R32)
        Z = Pr32*R32

        old_RZ = RZ
        RZ = Float32(frob_inner_prod(R32,Z))

        i += 1
        R_norm[i+1] = rnorm32

        println("PCG fully F32 it = $i, err = $(R_norm[i+1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if rnorm32 < tol32
            break
        end

        if old_RZ == 0f0 || !isfinite(old_RZ)
            @warn "PCG stopped because previous (R,Z)_F is invalid."
            break
        end

        beta = RZ/old_RZ
        if !isfinite(beta)
            @warn "PCG stopped because beta is not finite."
            break
        end

        P = Z + beta.*P
        AP = A32*P
    end

    return M32, R_norm[1:i+1]
end