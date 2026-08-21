using SparseArrays
using LinearAlgebra


function my_lopmr_lowprec_spgemm(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing
)
    n = size(A, 1)


    A64 = SparseMatrixCSC{Float64, Int}(A)
    Pr64 = SparseMatrixCSC{Float64, Int}(Pr)
    M64 = SparseMatrixCSC{Float64, Int}(M)


    A32 = SparseMatrixCSC{Float32, Int}(A64)

    Id = spdiagm(0 => ones(Float64, n))

    R_norm = zeros(Float64, itmax + 1)


    # Timing accumulators

    # A*Z
    time_Z_to_32 = 0.0
    time_AZ_float32 = 0.0
    time_AZ_to_64 = 0.0

    # A*P
    time_P_to_32 = 0.0
    time_AP_float32 = 0.0
    time_AP_to_64 = 0.0

    # A*M
    time_M_to_32 = 0.0
    time_AM_float32 = 0.0
    time_AM_to_64 = 0.0

    # These deliberately remain Float64
    time_PrAZ_float64 = 0.0
    time_PrAP_float64 = 0.0


    # Initial A*M in Float32

    time_M_to_32 += @elapsed begin
    M32 = Float32.(M64)
    end

    time_AM_float32 += @elapsed begin
    AM32 = A32 * M32
    end

    time_AM_to_64 += @elapsed begin
    AM64 = Float64.(AM32)
    end


    # Back to Float64

    R = Id - AM64

    Z = Pr64 * R

    # Previous locally optimal direction
    P = spzeros(Float64, n, n)


    i = 0

    R_norm[1] = frob_norm(R)

    println("my_lopmr_lowprec_spgemm it = 0, ", "err = $(R_norm[1]), ", "nnz(M)/n^2 = $(nnz(M64) / n^2)")


    # LOPMR iteration

    while i < itmax

    # A*Z in Float32

    time_Z_to_32 += @elapsed begin
    Z32 = Float32.(Z)
end

    time_AZ_float32 += @elapsed begin
    AZ32 = A32 * Z32
end

    time_AZ_to_64 += @elapsed begin
    AZ = Float64.(AZ32)
    end


    # Pr*AZ in Float64

    time_PrAZ_float64 += @elapsed begin
    PrAZ = Pr64 * AZ
    end


    # Computed delta and gamma

    if i == 0

    denominator = frob_inner_prod(AZ, PrAZ)

    if !isfinite(denominator) || abs(denominator) <= eps(Float64)

    println("Stopping: invalid initial denominator.")
    break
end


    delta = frob_inner_prod(Z, AZ) / denominator

    gamma = 1.0

    else

    # A*P in Float32

    time_P_to_32 += @elapsed begin
    P32 = Float32.(P)
    end

    time_AP_float32 += @elapsed begin
    AP32 = A32 * P32
    end

    time_AP_to_64 += @elapsed begin
    AP = Float64.(AP32)
    end


    # Pr*AP remains Float64

    time_PrAP_float64 += @elapsed begin
    PrAP = Pr64 * AP
    end


    # All Frobenius products remain Float64

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


    delta =(AP_PrAP * Z_AZ - AZ_PrAP * Z_AP) / c


    gamma =(AZ_PrAZ * Z_AP -AZ_PrAP * Z_AZ) / c
end


    if !isfinite(delta) || !isfinite(gamma)

    println("Stopping: delta or gamma is not finite.")
    break
    end


    if abs(delta) <= eps(Float64)

    println("Stopping: delta is nearly zero.")
    break
    end


    # M update remains Float64

    M64 = M64 + delta .* Z + gamma .* P


    # Optional symmetric dropping
    # Same dropping as Float64 implementation.

    if max_per_column !== nothing
    M64 = drop_largest_per_column(M64, max_per_column; keep_diagonal = true)
    end


    time_M_to_32 += @elapsed begin
    M32 = Float32.(M64)
    end

    time_AM_float32 += @elapsed begin
    AM32 = A32 * M32
    end

    time_AM_to_64 += @elapsed begin
    AM64 = Float64.(AM32)
    end


    # Everything here is Float64

    R = Id - AM64

    # Search-direction update remains Float64

    P = Z + (gamma / delta) .* P


    i += 1

    R_norm[i + 1] = frob_norm(R)


    println("my_lopmr_lowprec_spgemm it = $i, ", "err = $(R_norm[i + 1]), ", "nnz(M)/n^2 = $(nnz(M64) / n^2)")


    if R_norm[i + 1] < tol
    break
    end


    # Preconditioner application remains Float64

    Z = Pr64 * R
    end


    # Timing summary

    timings = (

    # A*Z
    Z_to_Float32 = time_Z_to_32,

    AZ_Float32_SpGEMM = time_AZ_float32,

    AZ_to_Float64 = time_AZ_to_64,


    # A*P
    P_to_Float32 = time_P_to_32,

    AP_Float32_SpGEMM = time_AP_float32,

    AP_to_Float64 = time_AP_to_64,


    # A*M
    M_to_Float32 = time_M_to_32,

    AM_Float32_SpGEMM = time_AM_float32,

    AM_to_Float64 = time_AM_to_64,


    # Deliberately Float64
    PrAZ_Float64_SpGEMM = time_PrAZ_float64,

    PrAP_Float64_SpGEMM = time_PrAP_float64,


    # Precision-conversion overhead
    total_conversion = time_Z_to_32 + time_AZ_to_64 + time_P_to_32 + time_AP_to_64 + time_M_to_32 + time_AM_to_64,


    # ONLY the three intentionally reduced-precision SpGEMMs
    total_Float32_SpGEMM = time_AZ_float32 + time_AP_float32 + time_AM_float32,


    # Preconditioner products kept in Float64
    total_preconditioner_SpGEMM = time_PrAZ_float64 + time_PrAP_float64)


    return M64, R_norm[1:i+1], timings
end