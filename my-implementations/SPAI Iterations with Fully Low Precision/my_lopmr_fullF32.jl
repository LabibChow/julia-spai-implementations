using SparseArrays
using LinearAlgebra

function my_lopmr_fully_float32(A, Pr, M, itmax, tol;
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
    P = spzeros(Float32,n,n)

    R_norm[1] = frob_norm(R32)

    println("LOPMR fully F32 it = 0, err = $(R_norm[1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

    i = 0
    while i < itmax
        AZ = A32*Z
        PrAZ = Pr32*AZ

        if i == 0
            num = Float32(frob_inner_prod(Z,AZ))
            den = Float32(frob_inner_prod(AZ,PrAZ))

            if den == 0f0 || !isfinite(den)
                @warn "LOPMR stopped because initial denominator is invalid."
                break
            end

            delta = num/den
            gamma = 1.0f0
        else
            AP = A32*P
            PrAP = Pr32*AP

            a = Float32(frob_inner_prod(AZ,PrAZ))
            b = Float32(frob_inner_prod(AP,PrAP))
            c1 = Float32(frob_inner_prod(AZ,PrAP))
            d = Float32(frob_inner_prod(Z,AZ))
            e = Float32(frob_inner_prod(Z,AP))

            c = a*b - c1*c1

            if c == 0f0 || !isfinite(c)
                @warn "LOPMR stopped because coefficient denominator is invalid."
                break
            end

            delta = (b*d - c1*e)/c
            gamma = (a*e - c1*d)/c
        end

        if !isfinite(delta) || !isfinite(gamma) || delta == 0f0
            @warn "LOPMR stopped because delta/gamma is invalid."
            break
        end

        M32 = M32 + delta.*Z + gamma.*P

        if max_per_column !== nothing
            M32 = drop_largest_per_column(M32,max_per_column; keep_diagonal=true)
        end

        R32 = Id32 - A32*M32
        rnorm32 = frob_norm(R32)

        P = Z + (gamma/delta).*P
        Z = Pr32*R32

        i += 1
        R_norm[i+1] = rnorm32

        println("LOPMR fully F32 it = $i, err = $(R_norm[i+1]), nnz(M)/n^2 = $(nnz(M32)/n^2)")

        if rnorm32 < tol32
            break
        end
    end

    return M32, R_norm[1:i+1]
end