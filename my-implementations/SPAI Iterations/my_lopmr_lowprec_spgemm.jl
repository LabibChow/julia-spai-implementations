using SparseArrays
using LinearAlgebra


function my_lopmr_lowprec_spgemm(A, Pr, M, itmax, tol)
    n = size(A, 1)

    # Main matrices remain in Float64
    A64 = SparseMatrixCSC{Float64, Int}(A)
    Pr64 = SparseMatrixCSC{Float64, Int}(Pr)
    M64 = SparseMatrixCSC{Float64, Int}(M)

    # A is stored once in Float32 for the low-precision SpGEMMs
    A32 = SparseMatrixCSC{Float32, Int}(A64)

    Id = spdiagm(0 => ones(Float64, n))

    R_norm = zeros(Float64, itmax + 1)

    # Timing accumulators
    time_Z_to_32 = 0.0
    time_AZ_float32 = 0.0
    time_AZ_to_64 = 0.0

    time_P_to_32 = 0.0
    time_AP_float32 = 0.0
    time_AP_to_64 = 0.0

    time_M_to_32 = 0.0
    time_AM_float32 = 0.0
    time_AM_to_64 = 0.0

    time_PrAZ_float64 = 0.0
    time_PrAP_float64 = 0.0

    # Initial actual residual:
    # R = I - A*M
    #
    # A*M is computed in Float32
    time_M_to_32 += @elapsed M32 = Float32.(M64)
    time_AM_float32 += @elapsed AM32 = A32 * M32
    time_AM_to_64 += @elapsed AM64 = Float64.(AM32)

    R = Id - AM64
    Z = Pr64 * R

    P = spzeros(Float64, n, n)

    i = 0
    R_norm[1] = frob_norm(R)

    println(
        "my_lopmr_lowprec_spgemm it = 0, " *
        "err = $(R_norm[1]), " *
        "nnz(M)/n^2 = $(nnz(M64) / n^2)"
    )

    while i < itmax
        # A*Z in Float32
        time_Z_to_32 += @elapsed Z32 = Float32.(Z)
        time_AZ_float32 += @elapsed AZ32 = A32 * Z32
        time_AZ_to_64 += @elapsed AZ = Float64.(AZ32)

        # Pr*AZ remains in Float64
        time_PrAZ_float64 += @elapsed PrAZ = Pr64 * AZ

        if i == 0
            denominator = frob_inner_prod(AZ, PrAZ)

            if !isfinite(denominator) ||
               abs(denominator) <= eps(Float64)

                println("Stopping: invalid initial denominator.")
                break
            end

            delta = frob_inner_prod(Z, AZ) / denominator
            gamma = 1.0
        else

            # A*P in Float32
            time_P_to_32 += @elapsed P32 = Float32.(P)
            time_AP_float32 += @elapsed AP32 = A32 * P32
            time_AP_to_64 += @elapsed AP = Float64.(AP32)

            # Pr*AP remains in Float64
            time_PrAP_float64 += @elapsed PrAP = Pr64 * AP

            AZ_PrAZ = frob_inner_prod(AZ, PrAZ)
            AP_PrAP = frob_inner_prod(AP, PrAP)
            AZ_PrAP = frob_inner_prod(AZ, PrAP)
            Z_AZ = frob_inner_prod(Z, AZ)
            Z_AP = frob_inner_prod(Z, AP)

            c = AZ_PrAZ * AP_PrAP - AZ_PrAP^2

            if !isfinite(c) || abs(c) <= eps(Float64)
                println("Stopping: invalid or nearly zero coefficient c.")
                break
            end

            delta =
                (AP_PrAP * Z_AZ - AZ_PrAP * Z_AP) / c

            gamma =
                (AZ_PrAZ * Z_AP - AZ_PrAP * Z_AZ) / c
        end

        if !isfinite(delta) || !isfinite(gamma)
            println("Stopping: delta or gamma is not finite.")
            break
        end

        if abs(delta) <= eps(Float64)
            println("Stopping: delta is nearly zero.")
            break
        end

        # Float64 iterate update
        M64 = M64 + delta .* Z + gamma .* P

        # Actual residual:
        # R = I - A*M
        #
        # A*M is computed in Float32
        time_M_to_32 += @elapsed M32 = Float32.(M64)
        time_AM_float32 += @elapsed AM32 = A32 * M32
        time_AM_to_64 += @elapsed AM64 = Float64.(AM32)

        R = Id - AM64

        # Float64 search-direction update
        P = Z + (gamma / delta) .* P

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println(
            "my_lopmr_lowprec_spgemm it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M64) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        # Preconditioner application remains in Float64
        Z = Pr64 * R
    end

    timings = (
        Z_to_Float32 = time_Z_to_32,
        AZ_Float32_SpGEMM = time_AZ_float32,
        AZ_to_Float64 = time_AZ_to_64,

        P_to_Float32 = time_P_to_32,
        AP_Float32_SpGEMM = time_AP_float32,
        AP_to_Float64 = time_AP_to_64,

        M_to_Float32 = time_M_to_32,
        AM_Float32_SpGEMM = time_AM_float32,
        AM_to_Float64 = time_AM_to_64,

        PrAZ_Float64_SpGEMM = time_PrAZ_float64,
        PrAP_Float64_SpGEMM = time_PrAP_float64,

        total_conversion =
            time_Z_to_32 +
            time_AZ_to_64 +
            time_P_to_32 +
            time_AP_to_64 +
            time_M_to_32 +
            time_AM_to_64,

        total_Float32_SpGEMM =
            time_AZ_float32 +
            time_AP_float32 +
            time_AM_float32,

        total_preconditioner_SpGEMM =
            time_PrAZ_float64 +
            time_PrAP_float64
    )

    return M64, R_norm[1:i+1], timings
end