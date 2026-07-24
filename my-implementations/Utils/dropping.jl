using SparseArrays


"""
    drop_largest_per_column(M, max_per_column; keep_diagonal=true)

Returns a sparse matrix in which each column retains at most
`max_per_column` entries of largest absolute magnitude.

The diagonal entry is preserved when `keep_diagonal=true`, provided that
the column has a valid diagonal position.
"""
function drop_largest_per_column(
    M::SparseMatrixCSC{T, Ti},
    max_per_column::Int;
    keep_diagonal::Bool = true,
) where {T<:AbstractFloat, Ti<:Integer}

    max_per_column >= 1 ||
        throw(ArgumentError("max_per_column must be at least 1."))

    nrows, ncols = size(M)

    output_rows = Int[]
    output_cols = Int[]
    output_values = T[]

    sizehint!(
        output_rows,
        min(nnz(M), max_per_column * ncols),
    )
    sizehint!(
        output_cols,
        min(nnz(M), max_per_column * ncols),
    )
    sizehint!(
        output_values,
        min(nnz(M), max_per_column * ncols),
    )

    M_rows = rowvals(M)
    M_values = nonzeros(M)

    for j in 1:ncols
        positions = collect(nzrange(M, j))

        isempty(positions) && continue

        rows_j = M_rows[positions]
        values_j = M_values[positions]

        number_to_keep = min(
            max_per_column,
            length(positions),
        )

        # Indices of the largest values by absolute magnitude.
        selected = partialsortperm(
            abs.(values_j),
            1:number_to_keep;
            rev = true,
        )

        if keep_diagonal && j <= nrows
            diagonal_local_position = findfirst(==(j), rows_j)

            if diagonal_local_position !== nothing &&
               diagonal_local_position ∉ selected

                # Replace the smallest selected entry with the diagonal.
                selected_magnitudes = abs.(values_j[selected])
                smallest_selected =
                    argmin(selected_magnitudes)

                selected[smallest_selected] =
                    diagonal_local_position
            end
        end

        # Sort by row so the resulting sparse structure is orderly.
        sort!(
            selected;
            by = index -> rows_j[index],
        )

        for local_index in selected
            value = values_j[local_index]

            if !iszero(value)
                push!(output_rows, rows_j[local_index])
                push!(output_cols, j)
                push!(output_values, value)
            end
        end
    end

    return sparse(
        output_rows,
        output_cols,
        output_values,
        nrows,
        ncols,
    )
end