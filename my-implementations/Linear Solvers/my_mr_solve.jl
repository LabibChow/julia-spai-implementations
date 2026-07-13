using SparseArrays
using LinearAlgebra

function my_mr_solve(A::AbstractMatrix{T},
                     b::Vector{T},
                     x0::Vector{T},
                     itmax::Int,
                     tol::Float64) where {T <: AbstractFloat}

    # MR solves Ax = b by minimizing the residual over x + span(r).
    #
    # r     = b - A*x
    # alpha = <A*r, r> / <A*r, A*r>
    # x     = x + alpha*r

    x = copy(x0)

    A64 = Float64.(A)
    b64 = Float64.(b)

    b_norm64 = norm(b64)
    if b_norm64 == 0.0
        b_norm64 = 1.0
    end

    residuals = zeros(Float64, itmax + 1)

    r = b - A * x

    residuals[1] =
        norm(b64 - A64 * Float64.(x)) / b_norm64

    i = 0

    while i < itmax

        Ar = A * r

        denominator = dot(Ar, Ar)

        if denominator == zero(T)
            println("my_mr_solve stopped: denominator became zero.")
            break
        end

        alpha = dot(Ar, r) / denominator

        x .+= alpha .* r
        r .-= alpha .* Ar

        i += 1

        residuals[i + 1] =
            norm(b64 - A64 * Float64.(x)) / b_norm64

        if residuals[i + 1] < tol
            break
        end
    end

    return x, residuals[1:i+1]
end