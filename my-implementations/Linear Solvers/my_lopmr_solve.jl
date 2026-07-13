using SparseArrays
using LinearAlgebra

function my_lopmr_solve(A::AbstractMatrix{T},
                        b::Vector{T},
                        x0::Vector{T},
                        M::AbstractMatrix{T},
                        itmax::Int,
                        tol::Float64) where {T <: AbstractFloat}

    # LOPMR solves Ax = b using a preconditioned locally optimal MR step.
    #
    # r = b - A*x
    # z = M*r
    #
    # It minimizes over:
    # x_k + span(z_k, previous_direction)

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

    n = length(b)

    s_prev = zeros(T, n)
    As_prev = zeros(T, n)
    has_prev = false

    i = 0

    while i < itmax

        z = M * r
        Az = A * z

        if !has_prev
            denominator = dot(Az, Az)

            if denominator == zero(T)
                println("my_lopmr_solve stopped: denominator became zero.")
                break
            end

            alpha = dot(Az, r) / denominator

            s = alpha .* z
            As = alpha .* Az

        else
            g11 = dot(Az, Az)
            g12 = dot(Az, As_prev)
            g22 = dot(As_prev, As_prev)

            h1 = dot(Az, r)
            h2 = dot(As_prev, r)

            detG = g11 * g22 - g12 * g12

            if detG == zero(T)
                # Fallback to PMR step
                denominator = dot(Az, Az)

                if denominator == zero(T)
                    println("my_lopmr_solve stopped: fallback denominator became zero.")
                    break
                end

                alpha = dot(Az, r) / denominator
                beta = zero(T)
            else
                alpha = (h1 * g22 - h2 * g12) / detG
                beta = (g11 * h2 - g12 * h1) / detG
            end

            s = alpha .* z .+ beta .* s_prev
            As = alpha .* Az .+ beta .* As_prev
        end

        x .+= s
        r .-= As

        s_prev .= s
        As_prev .= As
        has_prev = true

        i += 1

        residuals[i + 1] =
            norm(b64 - A64 * Float64.(x)) / b_norm64

        if residuals[i + 1] < tol
            break
        end
    end

    return x, residuals[1:i+1]
end