using SparseArrays
using LinearAlgebra


function my_pcg_lowprec_ap(A, Pr, M, itmax, tol)
    n = size(A, 1)

    # Main data remain in Float64
    A64 = SparseMatrixCSC{Float64, Int}(A)
    Pr64 = SparseMatrixCSC{Float64, Int}(Pr)
    M64 = SparseMatrixCSC{Float64, Int}(M)

    # A is stored once in Float32 for A * P
    A32 = SparseMatrixCSC{Float32, Int}(A64)

    Id = spdiagm(0 => ones(Float64, n))

    R_norm = zeros(Float64, itmax + 1)

    # Timing accumulators
    time_P_to_32 = 0.0
    time_AP_float32 = 0.0
    time_AP_to_64 = 0.0
    time_AM_float64 = 0.0

    # Initial residual in Float64
    AM = spzeros(Float64, n, n)

    time_AM_float64 += @elapsed begin
        AM = A64 * M64
    end

    R = Id - AM
    Z = Pr64 * R

    R_Z_frob = frob_inner_prod(R, Z)

    P = copy(Z)

    # Initial A * P in Float32
    P32 = spzeros(Float32, n, n)

    time_P_to_32 += @elapsed begin
        P32 = Float32.(P)
    end

    AP32 = spzeros(Float32, n, n)

    time_AP_float32 += @elapsed begin
        AP32 = A32 * P32
    end

    AP = spzeros(Float64, n, n)

    time_AP_to_64 += @elapsed begin
        AP = Float64.(AP32)
    end

    i = 0
    R_norm[1] = frob_norm(R)

    println(
        "my_pcg_lowprec_ap it = 0, " *
        "err = $(R_norm[1]), " *
        "nnz(M)/n^2 = $(nnz(M64) / n^2)"
    )

    while i < itmax
        P_AP_frob = frob_inner_prod(P, AP)

        if !isfinite(P_AP_frob) || abs(P_AP_frob) <= eps(Float64)
            println("Stopping: invalid or nearly zero (P, AP)_F.")
            break
        end

        alpha = R_Z_frob / P_AP_frob

        if !isfinite(alpha)
            println("Stopping: alpha is not finite.")
            break
        end

        # Float64 update
        M64 = M64 + alpha .* P

        # Actual residual computed fully in Float64
        # R = I - A64 * M64
        time_AM_float64 += @elapsed begin
            AM = A64 * M64
        end

        R = Id - AM

        # Preconditioning remains Float64
        Z = Pr64 * R

        old_R_Z_frob = R_Z_frob
        R_Z_frob = frob_inner_prod(R, Z)

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println(
            "my_pcg_lowprec_ap it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M64) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        if !isfinite(old_R_Z_frob) ||
           abs(old_R_Z_frob) <= eps(Float64)

            println("Stopping: invalid previous (R, Z)_F.")
            break
        end

        beta = R_Z_frob / old_R_Z_frob

        if !isfinite(beta)
            println("Stopping: beta is not finite.")
            break
        end

        # Float64 search-direction update
        P = Z + beta .* P

        # Only A * P is computed in Float32
        time_P_to_32 += @elapsed begin
            P32 = Float32.(P)
        end

        time_AP_float32 += @elapsed begin
            AP32 = A32 * P32
        end

        time_AP_to_64 += @elapsed begin
            AP = Float64.(AP32)
        end
    end

    timings = (
        P_to_Float32 = time_P_to_32,
        AP_Float32_SpGEMM = time_AP_float32,
        AP_to_Float64 = time_AP_to_64,
        AM_Float64_SpGEMM = time_AM_float64,

        total_AP_operations =
            time_P_to_32 +
            time_AP_float32 +
            time_AP_to_64
    )

    return M64, R_norm[1:i+1], timings
end