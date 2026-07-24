using SparseArrays
using LinearAlgebra

"""
    lowprec_spgemm(A32, X64)

Compute A * X in Float32 and convert the result back to Float64.

"""
function lowprec_spgemm(A32, X64)
    start_time = time_ns()
    X32 = Float32.(X64)
    conversion_to_32_time = (time_ns() - start_time) / 1e9

    start_time = time_ns()
    Y32 = A32 * X32
    spgemm_time = (time_ns() - start_time) / 1e9

    start_time = time_ns()
    Y64 = Float64.(Y32)
    conversion_to_64_time = (time_ns() - start_time) / 1e9

    return Y64,
           conversion_to_32_time,
           spgemm_time,
           conversion_to_64_time
end


"""
    my_pcg_lowprec_spgemm(A, Pr, M, itmax, tol)

Mixed-precision PCG-SPAI experiment.

Precision distribution:
- A used in A*P and A*M: Float32
- P and M are converted temporarily to Float32
- A*P and A*M are computed in Float32
- products are converted back to Float64
- M, R, Z, P, inner products and coefficients remain Float64
"""
function my_pcg_lowprec_spgemm(A, Pr, M, itmax, tol)
    n = size(A, 1)

    # Keep the main quantities in Float64.
    A64 = SparseMatrixCSC{Float64, Int}(A)
    Pr64 = SparseMatrixCSC{Float64, Int}(Pr)
    M = SparseMatrixCSC{Float64, Int}(M)

    # Store A once in lower precision.
    A32 = SparseMatrixCSC{Float32, Int}(A64)

    Id = spdiagm(0 => ones(Float64, n))

    R_norm = zeros(Float64, itmax + 1)

    # Timing accumulators.
    time_P_to_32 = 0.0
    time_AP_spgemm = 0.0
    time_AP_to_64 = 0.0

    time_M_to_32 = 0.0
    time_AM_spgemm = 0.0
    time_AM_to_64 = 0.0

    # Initial actual residual:
    # R₀ = I - A*M₀
    #
    # A*M is computed in Float32.
    AM, t_to32, t_mul, t_to64 = lowprec_spgemm(A32, M)

    time_M_to_32 += t_to32
    time_AM_spgemm += t_mul
    time_AM_to_64 += t_to64

    R = Id - AM

    # Preconditioner application remains in Float64.
    Z = Pr64 * R

    R_Z_frob = frob_inner_prod(R, Z)

    P = copy(Z)

    # Initial A*P, computed in Float32.
    AP, t_to32, t_mul, t_to64 = lowprec_spgemm(A32, P)

    time_P_to_32 += t_to32
    time_AP_spgemm += t_mul
    time_AP_to_64 += t_to64

    i = 0
    R_norm[1] = frob_norm(R)

    println(
        "my_pcg_lowprec_spgemm it = 0, ",
        "err = $(R_norm[1]), ",
        "nnz(M)/n^2 = $(nnz(M) / n^2)"
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

        # M update remains in Float64.
        M = M + alpha .* P

        # Actual residual:
        # R = I - A*M
        #
        # A*M is computed in Float32.
        AM, t_to32, t_mul, t_to64 = lowprec_spgemm(A32, M)

        time_M_to_32 += t_to32
        time_AM_spgemm += t_mul
        time_AM_to_64 += t_to64

        R = Id - AM

        # High-precision preconditioner application.
        Z = Pr64 * R

        old_R_Z_frob = R_Z_frob
        R_Z_frob = frob_inner_prod(R, Z)

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println(
            "my_pcg_lowprec_spgemm it = $i, ",
            "err = $(R_norm[i + 1]), ",
            "nnz(M)/n^2 = $(nnz(M) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end

        if !isfinite(old_R_Z_frob) ||
           abs(old_R_Z_frob) <= eps(Float64)

            println("Stopping: previous (R, Z)_F is invalid or nearly zero.")
            break
        end

        beta = R_Z_frob / old_R_Z_frob

        if !isfinite(beta)
            println("Stopping: beta is not finite.")
            break
        end

        # Search-direction update remains in Float64.
        P = Z + beta .* P

        # A*P is computed in Float32.
        AP, t_to32, t_mul, t_to64 = lowprec_spgemm(A32, P)

        time_P_to_32 += t_to32
        time_AP_spgemm += t_mul
        time_AP_to_64 += t_to64
    end

    timings = (
        P_to_Float32 = time_P_to_32,
        AP_Float32_SpGEMM = time_AP_spgemm,
        AP_to_Float64 = time_AP_to_64,

        M_to_Float32 = time_M_to_32,
        AM_Float32_SpGEMM = time_AM_spgemm,
        AM_to_Float64 = time_AM_to_64,

        total_conversion =
            time_P_to_32 +
            time_AP_to_64 +
            time_M_to_32 +
            time_AM_to_64,

        total_lowprec_spgemm =
            time_AP_spgemm +
            time_AM_spgemm
    )

    return M, R_norm[1:i+1], timings
end