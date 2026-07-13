using LinearAlgebra

frob_inner_prod(X, Y) = dot(X, Y)
frob_norm_squared(X) = dot(X, X)
frob_norm(X) = sqrt(frob_norm_squared(X))