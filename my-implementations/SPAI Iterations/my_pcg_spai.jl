using SparseArrays
using LinearAlgebra


function my_pcg(
    A,
    Pr,
    M,
    itmax,
    tol;
    max_per_column::Union{Nothing, Int} = nothing
)
    n = size(A, 1)

    R = spzeros(n, n)
    Z = spzeros(n, n)
    P = spzeros(n, n)
    AP = spzeros(n, n)

    R_norm = zeros(itmax + 1)

    # Initial residual
    R .= I - A * M

    # Preconditioned residual
    Z .= Pr * R

    R_Z_frob = frob_inner_prod(R, Z)

    P .= Z
    AP .= A * P

    i = 0
    R_norm[i + 1] = frob_norm(R)

    println("my_pcg it = 0, ", "err = $(R_norm[1]), ", "nnz(M)/n^2 = $(nnz(M) / n^2)")

    while i < itmax

    # alpha computation

    P_AP_frob = frob_inner_prod(P, AP)

    alpha = R_Z_frob / P_AP_frob

    # Update SPAI

    M = M + alpha .* P


    # Optional symmetric dropping
    # Droppedd BEFORE recomputing the true residual.

    # Once entries of M are dropped,
    #     R_new = R_old - alpha * AP       
    # is no longer the exact residual corresponding to the dropped M.

    if max_per_column !== nothing
    M = drop_largest_per_column(M, max_per_column; keep_diagonal = true)
    end

    # Recomputed true residual using the dropped M

    R = I - A * M


    # Preconditioned residual

    Z = Pr * R

    old_R_Z_frob = R_Z_frob

    R_Z_frob = frob_inner_prod(R, Z)


    # Residual norm

    i += 1

    R_norm[i + 1] = frob_norm(R)

    println("my_pcg it = $i, ", "err = $(R_norm[i + 1]), ", "nnz(M)/n^2 = $(nnz(M) / n^2)")

    if R_norm[i + 1] < tol
    break
    end


    # Compute beta

    beta = R_Z_frob / old_R_Z_frob

    # Search direction

    P = Z + beta .* P

    AP = A * P
    end

    return M, R_norm[1:i+1]
end