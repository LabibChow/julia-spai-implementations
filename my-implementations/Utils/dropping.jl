using SparseArrays
using LinearAlgebra

"""
    drop_largest_per_column(
        M,
        max_per_column;
        keep_diagonal = true,
    )

Symmetrizes `M` and drops entries while retaining at most
`max_per_column` selected candidates per column.

An off-diagonal pair `(i, j)` and `(j, i)` is retained only when both
entries were selected by their respective columns. Therefore, the
returned matrix is symmetric.
"""
function drop_largest_per_column(
    M::SparseMatrixCSC{T, Ti},
    max_per_column::Int;
    keep_diagonal::Bool = true,
) where {T<:AbstractFloat, Ti<:Integer}

    max_per_column >= 1 ||
        throw(ArgumentError(
            "max_per_column must be at least 1.",
        ))

    nrows, ncols = size(M)

    nrows == ncols ||
        throw(ArgumentError(
            "Symmetric dropping requires a square matrix.",
        ))

    n = nrows


    M_symmetric = (M + sparse(transpose(M))) / T(2)

    droptol!(M_symmetric, eps(T))
    dropzeros!(M_symmetric)

    # selected_rows[j] contains the row indices selected
    # as candidates in column j.
    selected_rows = [Set{Int}() for _ in 1:n]

    M_rows = rowvals(M_symmetric)
    M_values = nonzeros(M_symmetric)

    # 2. largest-magnitude candidates in each column
    for j in 1:n
        positions = collect(nzrange(M_symmetric, j))

        isempty(positions) && continue

        rows_j = M_rows[positions]
        values_j = M_values[positions]

        number_to_keep = min(
            max_per_column,
            length(positions),
        )

        selected = partialsortperm(
            abs.(values_j),
            1:number_to_keep;
            rev = true,
        )

        # Preserve the diagonal, if present.
        if keep_diagonal
            diagonal_local_position = findfirst(==(j), rows_j)

            if diagonal_local_position !== nothing &&
               diagonal_local_position ∉ selected

                selected_magnitudes =
                    abs.(values_j[selected])

                smallest_selected =
                    argmin(selected_magnitudes)

                selected[smallest_selected] =
                    diagonal_local_position
            end
        end

        for local_index in selected
            push!(
                selected_rows[j],
                rows_j[local_index],
            )
        end
    end

    # 3. only mutually selected symmetric pairs are retained
    output_rows = Int[]
    output_cols = Int[]
    output_values = T[]

    sizehint!(
        output_rows,
        min(nnz(M_symmetric), max_per_column * n),
    )
    sizehint!(
        output_cols,
        min(nnz(M_symmetric), max_per_column * n),
    )
    sizehint!(
        output_values,
        min(nnz(M_symmetric), max_per_column * n),
    )

    for j in 1:n
        for position in nzrange(M_symmetric, j)
            i = M_rows[position]
            value = M_values[position]

            if i == j
                # Keep the diagonal if it was selected.
                if i in selected_rows[j]
                    push!(output_rows, i)
                    push!(output_cols, j)
                    push!(output_values, value)
                end
            else
                # Keep (i,j) only if:
                #   i was selected in column j, and
                #   j was selected in column i.
                mutually_selected =
                    (i in selected_rows[j]) &&
                    (j in selected_rows[i])

                if mutually_selected
                    push!(output_rows, i)
                    push!(output_cols, j)
                    push!(output_values, value)
                end
            end
        end
    end

    M_dropped = sparse(
        output_rows,
        output_cols,
        output_values,
        n,
        n,
    )

    dropzeros!(M_dropped)

    return M_dropped
end