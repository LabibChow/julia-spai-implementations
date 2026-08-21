using SparseArrays
using LinearAlgebra

function my_lopmr_float16(
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
    AZ = spzeros(Float16, n, n)
    AP = spzeros(Float16, n, n)
    PrAZ = spzeros(Float16, n, n)
    PrAP = spzeros(Float16, n, n)

    R_norm = zeros(Float64, itmax + 1)

    R64 .= I - A64 * Float64.(M16)

    R16 .= Float16.(R64)
    Z .= Pr16 * R16

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        AZ .= A16 * Z
        AP .= A16 * P

        PrAZ .= Pr16 * AZ
        PrAP .= Pr16 * AP

        if !all(isfinite, nonzeros(AZ)) ||
           !all(isfinite, nonzeros(AP)) ||
           !all(isfinite, nonzeros(PrAZ)) ||
           !all(isfinite, nonzeros(PrAP))

            @warn(
                "LOPMR Float16 stopped because a matrix product contains Inf or NaN.",
                iteration = i,
            )
            break
        end

        Z64 = Float64.(Z)
        AZ64 = Float64.(AZ)
        AP64 = Float64.(AP)
        PrAZ64 = Float64.(PrAZ)
        PrAP64 = Float64.(PrAP)

        if i == 0
            denominator = frob_inner_prod(
                AZ64,
                PrAZ64,
            )

            if denominator == 0.0 ||
               !isfinite(denominator)

                @warn(
                    "LOPMR Float16 stopped because the delta denominator is invalid.",
                    iteration = i,
                    denominator = denominator,
                )
                break
            end

            delta =
                frob_inner_prod(Z64, AZ64) /
                denominator

            gamma = 1.0
        else
            AZ_PrAZ = frob_inner_prod(
                AZ64,
                PrAZ64,
            )

            AP_PrAP = frob_inner_prod(
                AP64,
                PrAP64,
            )

            AZ_PrAP = frob_inner_prod(
                AZ64,
                PrAP64,
            )

            Z_AZ = frob_inner_prod(
                Z64,
                AZ64,
            )

            Z_AP = frob_inner_prod(
                Z64,
                AP64,
            )

            c =
                AZ_PrAZ * AP_PrAP -
                AZ_PrAP^2

            if c == 0.0 || !isfinite(c)
                @warn(
                    "LOPMR Float16 stopped because the 2×2 system is invalid.",
                    iteration = i,
                    determinant = c,
                )
                break
            end

            delta =
                (
                    AP_PrAP * Z_AZ -
                    AZ_PrAP * Z_AP
                ) / c

            gamma =
                (
                    AZ_PrAZ * Z_AP -
                    AZ_PrAP * Z_AZ
                ) / c
        end

        if !isfinite(delta) ||
           abs(delta) > floatmax(Float16)

            @warn(
                "LOPMR Float16 stopped because delta cannot be represented in Float16.",
                iteration = i,
                delta = delta,
            )
            break
        end

        if !isfinite(gamma) ||
           abs(gamma) > floatmax(Float16)

            @warn(
                "LOPMR Float16 stopped because gamma cannot be represented in Float16.",
                iteration = i,
                gamma = gamma,
            )
            break
        end

        delta16 = Float16(delta)
        gamma16 = Float16(gamma)

        if delta16 == 0 && delta != 0.0
            @warn(
                "LOPMR Float16 stopped because delta rounded to zero.",
                iteration = i,
                delta = delta,
            )
            break
        end

        M16 .+= delta16 .* Z
        M16 .+= gamma16 .* P

        if max_per_column !== nothing
            M16 = drop_largest_per_column(
                M16,
                max_per_column;
                keep_diagonal = true,
            )
        end

        if !all(isfinite, nonzeros(M16))
            @warn(
                "LOPMR Float16 stopped because M contains Inf or NaN.",
                iteration = i,
            )
            break
        end

        R64 .= I - A64 * Float64.(M16)

        if !all(isfinite, nonzeros(R64))
            @warn(
                "LOPMR Float16 stopped because the Float64 residual is invalid.",
                iteration = i,
            )
            break
        end

        R16 .= Float16.(R64)

        if !all(isfinite, nonzeros(R16))
            @warn(
                "LOPMR Float16 stopped because the Float16 residual is invalid.",
                iteration = i,
            )
            break
        end

        direction_coefficient = gamma / delta

        if !isfinite(direction_coefficient) ||
           abs(direction_coefficient) > floatmax(Float16)

            @warn(
                "LOPMR Float16 stopped because gamma/delta cannot be represented in Float16.",
                iteration = i,
                coefficient = direction_coefficient,
            )
            break
        end

        direction16 = Float16(direction_coefficient)

        if direction16 == 0 &&
           direction_coefficient != 0.0

            @warn(
                "LOPMR Float16 stopped because gamma/delta rounded to zero.",
                iteration = i,
                coefficient = direction_coefficient,
            )
            break
        end

        P .*= direction16
        P .+= Z

        if !all(isfinite, nonzeros(P))
            @warn(
                "LOPMR Float16 stopped because P contains Inf or NaN.",
                iteration = i,
            )
            break
        end

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_lopmr_float16 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M16) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        Z .= Pr16 * R16

        if !all(isfinite, nonzeros(Z))
            @warn(
                "LOPMR Float16 stopped because Z contains Inf or NaN.",
                iteration = i,
            )
            break
        end
    end

    return M16, R_norm[1:i+1]
end