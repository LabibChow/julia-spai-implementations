using LinearAlgebra
using SparseArrays

function check_spd(A::AbstractMatrix; name = "A", symmetry_tolerance = 1e-12)
    
    # 1. Relative symmetry error
    # For matrices, norm(A) gives the Frobenius norm.

    A_norm = norm(A)

    symmetry_error =
        A_norm == 0.0 ?
        0.0 :
        norm(A - A') / A_norm

    # 2. Eigenvalue check

    A_dense = Matrix{Float64}(A)
    A_symmetric = Symmetric(A_dense)

    eigenvalues = eigvals(A_symmetric)

    lambda_min = minimum(eigenvalues)
    lambda_max = maximum(eigenvalues)

    # 3. Cholesky check

    cholesky_success = try
        cholesky(A_symmetric)
        true
    catch err
        println("Cholesky failed:")
        println(err)
        false
    end

    # 4. Final SPD classification

    # Float64-generated matrices can usually use 1e-12.
    # Float32-generated matrices should use a looser tolerance, like 1e-6

    is_spd =
        symmetry_error < symmetry_tolerance &&
        lambda_min > 0.0 &&
        cholesky_success

    # 5. Results

    println("SPD check for: ", name)
    println("Size: ", size(A))
    println("Symmetry error: ", symmetry_error)
    println("Smallest eigenvalue: ", lambda_min)
    println("Largest eigenvalue: ", lambda_max)
    println("Cholesky succeeded: ", cholesky_success)
    println("Classified as SPD: ", is_spd)

    return (
        symmetry_error = symmetry_error,
        lambda_min = lambda_min,
        lambda_max = lambda_max,
        cholesky_success = cholesky_success,
        is_spd = is_spd,
        eigenvalues = eigenvalues
    )
end