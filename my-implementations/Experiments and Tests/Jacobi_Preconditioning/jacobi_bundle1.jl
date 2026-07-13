using MatrixDepot 
using SparseArrays
using LinearAlgebra
using Plots
using Printf
import Measures

include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

# My SPAI implementations
spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")

include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))

# Matrix
matrix_name = "bundle1"

md = mdopen("Lourakis/bundle1")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("Matrix: ", matrix_name)
println("Size: ", size(A))
println("Nonzeros: ", nnz(A))
println("Symmetry error: ", norm(A - A'))

M0 = spdiagm(0 => ones(n))

itmax = Dict(
    "pmr"   => 200,
    "pcg"   => 200,
    "lopmr" => 200
)
tol = 1e-15

# Preconditioners

d = diag(A)

if any(iszero, d)
    error("Jacobi preconditioner cannot be formed because A has a zero diagonal entry.")
end

Pr_jacobi   = spdiagm(0 => 1.0 ./ d)

# PMR

time_pmr_jac = @elapsed begin
    M_pmr_jac, R_pmr_jac =
        my_pmr(A, Pr_jacobi, copy(M0), itmax["pmr"], tol)
end

# PCG

time_pcg_jac = @elapsed begin
    M_pcg_jac, R_pcg_jac =
        my_pcg(A, Pr_jacobi, copy(M0), itmax["pcg"], tol)
end

# LOPMR

time_lopmr_jac = @elapsed begin
    M_lopmr_jac, R_lopmr_jac =
        my_lopmr(A, Pr_jacobi, copy(M0), itmax["lopmr"], tol)
end


function print_summary(name, R_jac, time_jac)
    println("\n$name")
    println("-----------------------------------")
    println("Jacobi final residual:   ", R_jac[end])
    println("Jacobi minimum:          ", minimum(R_jac))
    println("Jacobi runtime:          ", time_jac)
    println("Residual < 1:          ", minimum(R_jac) < 1)
end

print_summary("PMR", R_pmr_jac, time_pmr_jac)
print_summary("PCG", R_pcg_jac, time_pcg_jac)
print_summary("LOPMR", R_lopmr_jac, time_lopmr_jac)


iteration_axis(R) = 0:(length(R) - 1)

p = plot(
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "$matrix_name — with Jacobi preconditioning",
    legend = :outerright,
    grid = true,
    size = (1200, 700),
    left_margin = 18 * Measures.mm,
    bottom_margin = 12 * Measures.mm,
    right_margin = 8 * Measures.mm
)

# PMR

plot!(p, iteration_axis(R_pmr_jac), R_pmr_jac,
      label = "PMR — Jacobi",
      color = :blue,
      linestyle = :solid,
      linewidth = 3)

# PCG

plot!(p, iteration_axis(R_pcg_jac), R_pcg_jac,
      label = "PCG — Jacobi",
      color = :red,
      linestyle = :solid,
      linewidth = 3)

# LOPMR

plot!(p, iteration_axis(R_lopmr_jac), R_lopmr_jac,
      label = "LOPMR — Jacobi",
      color = :green,
      linestyle = :solid,
      linewidth = 3)

display(p)

mkpath("jacobi_results")
savefig(
    p,
    joinpath(
        "jacobi_results",
        "jacobi_$(matrix_name).png"
    )
)