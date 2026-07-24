using SparseArrays
using LinearAlgebra


"""
    refine_spai_pattern(A, M_pattern; keep_explicit_zeros=false)

Recomputes the numerical values of an SPAI in Float64 while keeping the
sparsity pattern of `M_pattern` fixed.

For each column j, it solves

    min_x ‖e_j - A[:, J_j]x‖₂,
where J_j contains the permitted nonzero positions in column j of
`M_pattern`.

Arguments
---------
A: Sparse system matrix.

M_pattern: SPAI whose stored nonzero positions define the fixed sparsity pattern. Its values may be Float32 or Float64.

Keyword arguments
-----------------
keep_explicit_zeros:
    If true, numerically zero least-squares coefficients remain stored.
    If false, explicit zeros are removed from the final sparse matrix.

Returns
-------
M_refined:
    Float64 sparse approximate inverse with the same permitted pattern.
"""

function refine_spai_pattern(
    A::SparseMatrixCSC,
    M_pattern::SparseMatrixCSC;
    keep_explicit_zeros::Bool = false,
)
    n, m = size(A)

    if n != m
        throw(ArgumentError("A must be square, but size(A) = $(size(A))."))
    end

    if size(M_pattern) != (n, n)
        throw(
            DimensionMismatch(
                "M_pattern must have size ($n, $n), " *
                "but size(M_pattern) = $(size(M_pattern)).",
            ),
        )
    end

    # All least-squares calculations are performed in Float64.
    A64 = SparseMatrixCSC{Float64, Int}(A)

    # Triplets used to assemble the refined sparse matrix.
    row_indices = Int[]
    column_indices = Int[]
    values = Float64[]

    sizehint!(row_indices, nnz(M_pattern))
    sizehint!(column_indices, nnz(M_pattern))
    sizehint!(values, nnz(M_pattern))

    for j in 1:n
        # Stored row positions in column j of M_pattern.
        pattern_range = nzrange(M_pattern, j)
        J = collect(rowvals(M_pattern)[pattern_range])

        # Empty pattern produces an entirely zero column.
        isempty(J) && continue

        # Find rows touched by the selected columns of A.
        touched_rows = Int[]

        A_rows = rowvals(A64)

    for column_index in J
        for position in nzrange(A64, column_index)
        push!(touched_rows, A_rows[position])
        end
    end

        # Including row j represents the nonzero entry of e_j even when
        # none of the selected columns of A touches that row.
        push!(touched_rows, j)

        sort!(touched_rows)
        unique!(touched_rows)

        I = touched_rows

        # Reduced least-squares problem:
        #
        #     min_x ‖A[I, J]x - e_j[I]‖₂
        #
        A_local = A64[I, J]

        rhs = zeros(Float64, length(I))
        target_position = searchsortedfirst(I, j)
        rhs[target_position] = 1.0

        # Sparse least-squares solve in Float64.
        x = A_local \ rhs

        append!(row_indices, J)
        append!(column_indices, fill(j, length(J)))
        append!(values, x)
    end

    M_refined = sparse(
        row_indices,
        column_indices,
        values,
        n,
        n,
    )

    if !keep_explicit_zeros
        dropzeros!(M_refined)
    end

    return M_refined
end