using SparseArrays
using LinearAlgebra

function my_pmr_solve(A::AbstractMatrix{T},
                      b::Vector{T},
                      x0::Vector{T},
                      M::AbstractMatrix{T},
                      itmax::Int,
                      tol::Float64) where {T <: AbstractFloat}

    # PMR solves Ax = b using a preconditioner M ≈ A^(-1).
    #
    # r     = b - A*x
    # z     = M*r
    # alpha = <A*z, r> / <A*z, A*z>
    # x     = x + alpha*z

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

        z = M * r
        Az = A * z

        denominator = dot(Az, Az)

        if denominator == zero(T)
            println("my_pmr_solve stopped: denominator became zero.")
            break
        end

        alpha = dot(Az, r) / denominator

        x .+= alpha .* z
        r .-= alpha .* Az

        i += 1

        residuals[i + 1] =
            norm(b64 - A64 * Float64.(x)) / b_norm64

        if residuals[i + 1] < tol
            break
        end
    end

    return x, residuals[1:i+1]
end