using SparseArrays
using LinearAlgebra


function my_pmr_float32(A, Pr, M, itmax, tol; max_per_column::Union{Nothing, Int} = nothing,)
    A64 = Float64.(A)

    A32 = Float32.(A)
    Pr32 = Float32.(Pr)
    M32 = Float32.(M)

    n = size(A32, 1)

    R64 = spzeros(Float64, n, n)

    R32 = spzeros(Float32, n, n)
    Z = spzeros(Float32, n, n)
    PrA = spzeros(Float32, n, n)
    PrAZ = spzeros(Float32, n, n)

    R_norm = zeros(Float64, itmax + 1)

    PrA .= Pr32 * A32

    # Actual residual in Float64.
    R64 .= I - A64 * Float64.(M32)

    R32 .= Float32.(R64)
    Z .= Pr32 * R32
    PrAZ .= PrA * Z

    i = 0
    R_norm[i + 1] = frob_norm(R64)

    while i < itmax
        denominator = frob_norm_squared(PrAZ)

        if denominator == 0
            @warn "PMR stopped because the alpha denominator is zero."
            break
        end

        alpha = Float32(
            frob_inner_prod(Z, PrAZ) / denominator
        )

        # Update the approximate inverse.
        M32 .+= alpha .* Z

        # Restrict the sparsity immediately after the update.
        if max_per_column !== nothing
            M32 = drop_largest_per_column(
                M32,
                max_per_column;
                keep_diagonal = true,
            )
        end

        # Recompute the actual residual after dropping.
        R64 .= I - A64 * Float64.(M32)

        R32 .= Float32.(R64)
        Z .= Pr32 * R32
        PrAZ .= PrA * Z

        i += 1
        R_norm[i + 1] = frob_norm(R64)

        println(
            "my_pmr_float32 it = $i, " *
            "err = $(R_norm[i + 1]), " *
            "nnz(M)/n^2 = $(nnz(M32) / n^2)"
        )

        if R_norm[i + 1] < tol
            break
        end
    end

    return M32, R_norm[1:i+1]
end