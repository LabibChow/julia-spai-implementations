using SparseArrays
using LinearAlgebra


function my_pmr(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing
)
    n = size(A, 1)

    R_norm = zeros(Float64, itmax + 1)

    # Initial residual
    R = I - A * M

    # Preconditioned residual
    Z = Pr * R

    # PMR direction
    P = copy(Z)

    # Float64 A*P
    AP = A * P

    # Everything after A*P remains Float64
    PrAP = Pr * AP

    i = 0
    R_norm[1] = frob_norm(R)

    println("my_pmr it = 0, ", "err = $(R_norm[1]), ", "nnz(M)/n^2 = $(nnz(M) / n^2)")

    while i < itmax

    denominator = frob_norm_squared(PrAP)

    if denominator == 0.0 || !isfinite(denominator)
    @warn "PMR stopped because the alpha denominator is invalid."
    break
end

    alpha = frob_inner_prod(P, PrAP) / denominator

    if !isfinite(alpha)
    println("Stopping: alpha is not finite.")
    break
    end

    # Update SPAI
    M = M + alpha .* P

    # Optional symmetric dropping
    if max_per_column !== nothing
    M = drop_largest_per_column(M, max_per_column; keep_diagonal = true)
    end

    # True residual after dropping
    R = I - A * M

    # Float64 PMR quantities
    Z = Pr * R
    P = copy(Z)

    AP = A * P
    PrAP = Pr * AP

    i += 1

    R_norm[i + 1] = frob_norm(R)

    println("my_pmr it = $i, ", "err = $(R_norm[i + 1]), ", "nnz(M)/n^2 = $(nnz(M) / n^2)")

    if R_norm[i + 1] < tol
    break
    end
    
end

    return M, R_norm[1:i+1]
end