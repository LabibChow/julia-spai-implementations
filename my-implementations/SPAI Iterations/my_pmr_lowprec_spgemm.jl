using SparseArrays
using LinearAlgebra


function my_pmr_lowprec_spgemm(A, Pr, M, itmax, tol)
    n = size(A, 1)

    # Main matrices remain in Float64
    A64 = SparseMatrixCSC{Float64, Int}(A)
    Pr64 = SparseMatrixCSC{Float64, Int}(Pr)
    M64 = SparseMatrixCSC{Float64, Int}(M)

    # Float32 versions used only for SpGEMMs
    A32 = SparseMatrixCSC{Float32, Int}(A64)

    # Compute Pr*A once, then store a Float32 copy
    PrA64 = Pr64 * A64
    PrA32 = SparseMatrixCSC{Float32, Int}(PrA64)

    Id = spdiagm(0 => ones(Float64, n))
    R_norm = zeros(Float64, itmax + 1)

    # Timing
    time_M_to_32 = 0.0
    time_AM_float32 = 0.0
    time_AM_to_64 = 0.0

    time_Z_to_32 = 0.0
    time_PrAZ_float32 = 0.0
    time_PrAZ_to_64 = 0.0

    # Initial A*M in Float32
    time_M_to_32 += @elapsed M32 = Float32.(M64)
    time_AM_float32 += @elapsed AM32 = A32 * M32
    time_AM_to_64 += @elapsed AM64 = Float64.(AM32)

    R = Id - AM64

    # Preconditioner application remains Float64
    Z = Pr64 * R

    # Initial (Pr*A)*Z in Float32
    time_Z_to_32 += @elapsed Z32 = Float32.(Z)
    time_PrAZ_float32 += @elapsed PrAZ32 = PrA32 * Z32
    time_PrAZ_to_64 += @elapsed PrAZ = Float64.(PrAZ32)

    i = 0
    R_norm[1] = frob_norm(R)

    println(
        "my_pmr_lowprec_spgemm it = 0, " *
        "err = $(R_norm[1]), " *
        "nnz(M)/n^2 = $(nnz(M64) / n^2)"
    )

    while i < itmax
        denominator = frob_norm_squared(PrAZ)

        if !isfinite(denominator) || denominator <= eps(Float64)
            println("Stopping: invalid or nearly zero denominator.")
            break
        end

        alpha = frob_inner_prod(Z, PrAZ) / denominator

        if !isfinite(alpha)
            println("Stopping: alpha is not finite.")
            break
        end

        # Update remains Float64
        M64 = M64 + alpha .* Z

        # Actual residual: A*M in Float32
        time_M_to_32 += @elapsed M32 = Float32.(M64)
        time_AM_float32 += @elapsed AM32 = A32 * M32
        time_AM_to_64 += @elapsed AM64 = Float64.(AM32)

        R = Id - AM64

        # High-precision preconditioner application
        Z = Pr64 * R

        # (Pr*A)*Z in Float32
        time_Z_to_32 += @elapsed Z32 = Float32.(Z)
        time_PrAZ_float32 += @elapsed PrAZ32 = PrA32 * Z32
        time_PrAZ_to_64 += @elapsed PrAZ = Float64.(PrAZ32)

        i += 1
        R_norm[i + 1] = frob_norm(R)

        println(
            "my_pmr_lowprec_spgemm it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M64) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end
    end

    timings = (
        M_to_Float32 = time_M_to_32,
        AM_Float32_SpGEMM = time_AM_float32,
        AM_to_Float64 = time_AM_to_64,

        Z_to_Float32 = time_Z_to_32,
        PrAZ_Float32_SpGEMM = time_PrAZ_float32,
        PrAZ_to_Float64 = time_PrAZ_to_64,

        total_conversion =
            time_M_to_32 +
            time_AM_to_64 +
            time_Z_to_32 +
            time_PrAZ_to_64,

        total_Float32_SpGEMM =
            time_AM_float32 +
            time_PrAZ_float32
    )

    return M64, R_norm[1:i+1], timings
end