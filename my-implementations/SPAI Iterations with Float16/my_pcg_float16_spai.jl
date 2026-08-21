using SparseArrays
using LinearAlgebra

function my_pcg_float16(
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
    P = spzeros(Float16, n, n)
    AP = spzeros(Float16, n, n)

    R_norm = zeros(Float64, itmax + 1)

    R64 .= I - A64 * Float64.(M16)

    R16 .= Float16.(R64)
    Z .= Pr16 * R16

    R_Z_frob = frob_inner_prod(
        Float64.(R16),
        Float64.(Z),
    )

    P .= Z
    AP = A16 * P

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        P_AP_frob = frob_inner_prod(
            Float64.(P),
            Float64.(AP),
        )

        if P_AP_frob == 0.0 || !isfinite(P_AP_frob)
            @warn(
                "PCG Float16 stopped because the alpha denominator is invalid.",
                iteration = i,
                denominator = P_AP_frob,
            )
            break
        end

        alpha = R_Z_frob / P_AP_frob

        if !isfinite(alpha) ||
           abs(alpha) > floatmax(Float16)

            @warn(
                "PCG Float16 stopped because alpha cannot be represented in Float16.",
                iteration = i,
                alpha = alpha,
            )
            break
        end

        alpha16 = Float16(alpha)

        if alpha16 == 0 && alpha != 0.0
            @warn(
                "PCG Float16 stopped because alpha rounded to zero.",
                iteration = i,
                alpha = alpha,
            )
            break
        end

        M_candidate = M16 + alpha16 .* P

if max_per_column !== nothing
    M_candidate = drop_largest_per_column(
        M_candidate,
        max_per_column;
        keep_diagonal = true,
    )
end

if !all(isfinite, nonzeros(M_candidate))
    @warn(
        "PCG Float16 stopped because the update would produce Inf or NaN.",
        iteration = i,
    )
    break
end

M16 = M_candidate

        R64 .= I - A64 * Float64.(M16)

        if !all(isfinite, nonzeros(R64))
            @warn(
                "PCG Float16 stopped because the Float64 residual is invalid.",
                iteration = i,
            )
            break
        end

        R16 .= Float16.(R64)

        if !all(isfinite, nonzeros(R16))
            @warn(
                "PCG Float16 stopped because the Float16 residual is invalid.",
                iteration = i,
            )
            break
        end

        Z .= Pr16 * R16

        old_R_Z_frob = R_Z_frob

        R_Z_frob = frob_inner_prod(
            Float64.(R16),
            Float64.(Z),
        )

        if !isfinite(R_Z_frob)
            @warn(
                "PCG Float16 stopped because the residual inner product is invalid.",
                iteration = i,
                residual_inner_product = R_Z_frob,
            )
            break
        end

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_pcg_float16 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M16) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        if old_R_Z_frob == 0.0 ||
           !isfinite(old_R_Z_frob)

            @warn(
                "PCG Float16 stopped because the beta denominator is invalid.",
                iteration = i,
                denominator = old_R_Z_frob,
            )
            break
        end

        beta = R_Z_frob / old_R_Z_frob

        if !isfinite(beta) ||
           abs(beta) > floatmax(Float16)

            @warn(
                "PCG Float16 stopped because beta cannot be represented in Float16.",
                iteration = i,
                beta = beta,
            )
            break
        end

        beta16 = Float16(beta)

        if beta16 == 0 && beta != 0.0
            @warn(
                "PCG Float16 stopped because beta rounded to zero.",
                iteration = i,
                beta = beta,
            )
            break
        end

        P = Z + beta16 .* P

        if !all(isfinite, nonzeros(P))
            @warn(
                "PCG Float16 stopped because P contains Inf or NaN.",
                iteration = i,
            )
            break
        end

        AP = A16 * P

        if !all(isfinite, nonzeros(AP))
            @warn(
                "PCG Float16 stopped because AP contains Inf or NaN.",
                iteration = i,
            )
            break
        end
    end

    return M16, R_norm[1:i+1]
end