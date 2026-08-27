using SparseArrays
using LinearAlgebra

function my_pcg_fully_float16(A, Pr, M, itmax, tol;
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
    RZ = Float16(frob_inner_prod(R16,Z))
    P = copy(Z)
    AP = A16*P

    R_norm[1] = frob_norm(R16)

    println("PCG fully F16 it = 0, err = $(R_norm[1]), nnz(M)/n^2 = $(nnz(M16)/n^2)")

    i = 0
    while i < itmax
        PAP = Float16(frob_inner_prod(P,AP))

        if PAP == Float16(0) || !isfinite(PAP)
            @warn "PCG stopped because alpha denominator is invalid."
            break
        end

        alpha = RZ/PAP
        if !isfinite(alpha)
            @warn "PCG stopped because alpha is not finite."
            break
        end

        M16 = M16 + alpha.*P

        if max_per_column !== nothing
            M16 = drop_largest_per_column(M16,max_per_column; keep_diagonal=true)
        end

        R16 = Id16 - A16*M16
        rnorm16 = frob_norm(R16)
        Z = Pr16*R16

        old_RZ = RZ
        RZ = Float16(frob_inner_prod(R16,Z))

        i += 1
        R_norm[i+1] = rnorm16

        println("PCG fully F16 it = $i, err = $(R_norm[i+1]), nnz(M)/n^2 = $(nnz(M16)/n^2)")

        if rnorm16 < tol16
            break
        end

        if old_RZ == Float16(0) || !isfinite(old_RZ)
            @warn "PCG stopped because previous (R,Z)_F is invalid."
            break
        end

        beta = RZ/old_RZ
        if !isfinite(beta)
            @warn "PCG stopped because beta is not finite."
            break
        end

        P = Z + beta.*P
        AP = A16*P
    end

    return M16, R_norm[1:i+1]
end