using SparseArrays
using LinearAlgebra

function my_pmr_float16(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing,
)
    A64 = Float64.(A)

    A16 = Float16.(A)
    Pr16 = Float16.(Pr)
    M16 = Float16.(M)

    n = size(A16, 1)

    R64 = spzeros(Float64, n, n)

    R16 = spzeros(Float16, n, n)
    Z = spzeros(Float16, n, n)
    PrA = spzeros(Float16, n, n)
    PrAZ = spzeros(Float16, n, n)

    R_norm = zeros(Float64, itmax + 1)

    PrA .= Pr16 * A16

    R64 .= I - A64 * Float64.(M16)

    R16 .= Float16.(R64)
    Z .= Pr16 * R16
    PrAZ .= PrA * Z

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        Z64 = Float64.(Z)
        PrAZ64 = Float64.(PrAZ)

        denominator = frob_norm_squared(PrAZ64)

        if denominator == 0.0 || !isfinite(denominator)
            @warn(
                "PMR Float16 stopped because the alpha denominator is invalid.",
                iteration = i,
                denominator = denominator,
            )
            break
        end

        alpha =
            frob_inner_prod(Z64, PrAZ64) /
            denominator

        if !isfinite(alpha) ||
           abs(alpha) > floatmax(Float16)

            @warn(
                "PMR Float16 stopped because alpha cannot be represented in Float16.",
                iteration = i,
                alpha = alpha,
            )
            break
        end

        alpha16 = Float16(alpha)

        if alpha16 == 0 && alpha != 0.0
            @warn(
                "PMR Float16 stopped because alpha rounded to zero.",
                iteration = i,
                alpha = alpha,
            )
            break
        end

        M16 = M16 + alpha16 .* Z

        if max_per_column !== nothing
            M16 = drop_largest_per_column(
                M16,
                max_per_column;
                keep_diagonal = true,
            )
        end

        if !all(isfinite, nonzeros(M16))
            @warn(
                "PMR Float16 stopped because M contains Inf or NaN.",
                iteration = i,
            )
            break
        end

        R64 .= I - A64 * Float64.(M16)

        if !all(isfinite, nonzeros(R64))
            @warn(
                "PMR Float16 stopped because the Float64 residual is invalid.",
                iteration = i,
            )
            break
        end

        R16 .= Float16.(R64)

        if !all(isfinite, nonzeros(R16))
            @warn(
                "PMR Float16 stopped because the Float16 residual is invalid.",
                iteration = i,
            )
            break
        end

        Z .= Pr16 * R16
        PrAZ .= PrA * Z

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_pmr_float16 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M16) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end
    end

    return M16, R_norm[1:i+1]
end