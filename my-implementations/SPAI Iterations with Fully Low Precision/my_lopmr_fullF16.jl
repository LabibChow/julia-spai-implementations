using SparseArrays
using LinearAlgebra

function my_lopmr_fully_float16(A, Pr, M, itmax, tol;
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
    P = spzeros(Float16,n,n)

    R_norm[1] = frob_norm(R16)

    println("LOPMR fully F16 it = 0, err = $(R_norm[1]), nnz(M)/n^2 = $(nnz(M16)/n^2)")

    i = 0
    while i < itmax
        AZ = A16*Z
        PrAZ = Pr16*AZ

        if i == 0
            num = Float16(frob_inner_prod(Z,AZ))
            den = Float16(frob_inner_prod(AZ,PrAZ))

            if den == Float16(0) || !isfinite(den)
                @warn "LOPMR stopped because initial denominator is invalid."
                break
            end

            delta = num/den
            gamma = Float16(1)
        else
            AP = A16*P
            PrAP = Pr16*AP

            a = Float16(frob_inner_prod(AZ,PrAZ))
            b = Float16(frob_inner_prod(AP,PrAP))
            c1 = Float16(frob_inner_prod(AZ,PrAP))
            d = Float16(frob_inner_prod(Z,AZ))
            e = Float16(frob_inner_prod(Z,AP))

            c = a*b - c1*c1

            if c == Float16(0) || !isfinite(c)
                @warn "LOPMR stopped because coefficient denominator is invalid."
                break
            end

            delta = (b*d - c1*e)/c
            gamma = (a*e - c1*d)/c
        end

        if !isfinite(delta) || !isfinite(gamma) || delta == Float16(0)
            @warn "LOPMR stopped because delta/gamma is invalid."
            break
        end

        M16 = M16 + delta.*Z + gamma.*P

        if max_per_column !== nothing
            M16 = drop_largest_per_column(M16,max_per_column; keep_diagonal=true)
        end

        R16 = Id16 - A16*M16
        rnorm16 = frob_norm(R16)

        P = Z + (gamma/delta).*P
        Z = Pr16*R16

        i += 1
        R_norm[i+1] = rnorm16

        println("LOPMR fully F16 it = $i, err = $(R_norm[i+1]), nnz(M)/n^2 = $(nnz(M16)/n^2)")

        if rnorm16 < tol16
            break
        end
    end

    return M16, R_norm[1:i+1]
end