using SparseArrays
using LinearAlgebra

function my_cg_solve(A::AbstractMatrix{T},
                     b::Vector{T},
                     x0::Vector{T},
                     itmax::Int,
                     tol::Float64) where {T <: AbstractFloat}

    # CG solves Ax = b for symmetric positive definite A.
    #
    # A     = SPD matrix
    # b     = right-hand side vector
    # x0    = initial guess
    # itmax = maximum number of iterations
    # tol   = tolerance based on relative residual

    x = copy(x0)

    r = b - A * x
    p = copy(r)

    residuals = zeros(Float64, itmax + 1)

    # I have converted once for accurate residual calculations, to avoid repeated conversions during every iteration.
    A64 = Float64.(A)
    b64 = Float64.(b)

    b_norm64 = norm(b64)

    if b_norm64 == 0.0
        b_norm64 = 1.0
    end

    # Initial true relative residual computed in Float64.
    residuals[1] =
        norm(b64 - A64 * Float64.(x)) / b_norm64

    rsold = dot(r, r)

    i = 0

    while i < itmax
        Ap = A * p

        denom = dot(p, Ap)

        if denom == zero(T)
            println("my_cg_solve stopped: denominator became zero.")
            break
        end

        alpha = rsold / denom

        x .+= alpha .* p
        r .-= alpha .* Ap

        i += 1

        # True relative residual computed in Float64.
        residuals[i + 1] =
            norm(b64 - A64 * Float64.(x)) / b_norm64

        if residuals[i + 1] < tol
            break
        end

        rsnew = dot(r, r)

        if rsold == zero(T)
            println("my_cg_solve stopped: rsold became zero.")
            break
        end

        beta = rsnew / rsold

        p .= r .+ beta .* p

        rsold = rsnew
    end

    return x, residuals[1:i+1]
end