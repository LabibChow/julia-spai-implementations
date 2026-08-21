using SparseArrays
using LinearAlgebra


function my_lopmr(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing
)
    n = size(A, 1)

    # Keep main quantities in Float64
    A64 = SparseMatrixCSC{Float64, Int}(A)
    Pr64 = SparseMatrixCSC{Float64, Int}(Pr)
    M = SparseMatrixCSC{Float64, Int}(M)

    Id = spdiagm(
        0 => ones(Float64, n)
    )

    R_norm = zeros(Float64, itmax + 1)

    # Initial residual and preconditioned residual

    R = Id - A64 * M
    Z = Pr64 * R

    # Previous locally optimal direction.
    # At iteration 0 this is zero.
    P = spzeros(Float64, n, n)

    i = 0
    R_norm[1] = frob_norm(R)

    println(
        "my_lopmr it = 0, ",
        "err = $(R_norm[1]), ",
        "nnz(M)/n^2 = $(nnz(M) / n^2)"
    )


    # LOPMR iteration

    while i < itmax

    # A*Z and Pr*A*Z in Float64

    AZ = A64 * Z
    PrAZ = Pr64 * AZ


    # Computing locally optimal coefficients

    if i == 0

    denominator = frob_inner_prod(AZ, PrAZ)

    if !isfinite(denominator) ||
    abs(denominator) <= eps(Float64)

    println("Stopping: invalid initial denominator.")
    break
    end

    delta =
    frob_inner_prod(Z, AZ) / denominator

    gamma = 1.0

    else

    # A*P and Pr*A*P in Float64
    AP = A64 * P
    PrAP = Pr64 * AP

    AZ_PrAZ = frob_inner_prod(AZ, PrAZ)

    AP_PrAP = frob_inner_prod(AP, PrAP)

    AZ_PrAP = frob_inner_prod(AZ, PrAP)

    Z_AZ = frob_inner_prod(Z,AZ)

    Z_AP = frob_inner_prod(Z, AP)

    c = AZ_PrAZ * AP_PrAP - AZ_PrAP^2

    if c == 0.0 || !isfinite(c)

    println("Stopping: invalid or nearly zero coefficient c.")
    break
    end

    delta = (AP_PrAP * Z_AZ - AZ_PrAP * Z_AP) / c

    gamma = (AZ_PrAZ * Z_AP - AZ_PrAP * Z_AZ) / c
    end


    if !isfinite(delta) || !isfinite(gamma)

    println("Stopping: delta or gamma is not finite.")
    break
    end

    if abs(delta) <= eps(Float64)
    println("Stopping: delta is nearly zero.")
    break 
    end


    # Update M in Float64

    M = M + delta .* Z + gamma .* P


    # Optional symmetric dropping

    if max_per_column !== nothing
    M = drop_largest_per_column(M, max_per_column; keep_diagonal = true)
    end


    # True residual corresponding to dropped M

    R = Id - A64 * M


    # Search-direction update

    P = Z + (gamma / delta) .* P


    i += 1

    R_norm[i + 1] = frob_norm(R)

    println("my_lopmr it = $i, ", "err = $(R_norm[i + 1]), ", "nnz(M)/n^2 = $(nnz(M) / n^2)")


    if R_norm[i + 1] < tol  
    break
    end


    # New preconditioned residual for next iteration
    Z = Pr64 * R
    end


    return M,
    R_norm[1:i+1]
end