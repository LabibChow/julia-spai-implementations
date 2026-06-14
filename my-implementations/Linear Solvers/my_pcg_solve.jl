using SparseArrays
using LinearAlgebra

function my_pcg_solve(A::AbstractMatrix{T},
                      b::Vector{T},
                      x0::Vector{T},
                      M::AbstractMatrix{T},
                      itmax::Int,
                      tol::Float64) where {T <: AbstractFloat}

    # PCG solves Ax = b using a preconditioner M.
    #
    # A     = SPD matrix
    # b     = right-hand side vector
    # x0    = initial guess
    # M     = preconditioner, ideally M ≈ A^(-1)
    # itmax = maximum number of iterations
    # tol   = tolerance based on relative residual
    #
    # Since my SPAI gives an approximate inverse, I use:
    # z = M * r

    x = copy(x0)

    r = b - A * x
    z = M * r
    p = copy(z)

    residuals = zeros(Float64, itmax + 1)

    b_norm64 = norm(Float64.(b))

    if b_norm64 == 0.0
        b_norm64 = 1.0
    end

    # High-precision residual computation
    residuals[1] = norm(Float64.(r)) / b_norm64

    rzold = dot(r, z)

    i = 0

    while i < itmax
        Ap = A * p

        denom = dot(p, Ap)

        if denom == zero(T)
            println("my_pcg_solve stopped: denominator became zero.")
            break
        end

        alpha = rzold / denom

        x .+= alpha .* p
        r .-= alpha .* Ap

        i += 1

        # High-precision residual computation
        residuals[i + 1] = norm(Float64.(b) - Float64.(A) * Float64.(x)) / b_norm64

        #println("my_pcg_solve it = $i, rel_res = $(residuals[i + 1])")

        if residuals[i + 1] < tol
            break
        end

        z = M * r

        rznew = dot(r, z)

        if rzold == zero(T)
            println("my_pcg_solve stopped: rzold became zero.")
            break
        end

        beta = rznew / rzold

        p .= z .+ beta .* p

        rzold = rznew
    end

    return x, residuals[1:i+1]
end