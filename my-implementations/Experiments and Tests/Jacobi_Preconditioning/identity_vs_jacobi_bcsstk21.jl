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
matrix_name = "bcsstk21"

md = mdopen("HB/bcsstk21")
A = sparse(Float64.(md.A))

n = size(A, 1)

println("Matrix: ", matrix_name)
println("Size: ", size(A))
println("Nonzeros: ", nnz(A))
println("Symmetry error: ", norm(A - A'))

M0 = spdiagm(0 => ones(n))

itmax = 1000
tol = 1e-15

# Preconditioners

d = diag(A)

if any(iszero, d)
    error("Jacobi preconditioner cannot be formed because A has a zero diagonal entry.")
end

Pr_identity = spdiagm(0 => ones(n))
Pr_jacobi   = spdiagm(0 => 1.0 ./ d)

# PMR

time_pmr_id = @elapsed begin
    M_pmr_id, R_pmr_id =
        my_pmr(A, Pr_identity, copy(M0), itmax, tol)
end

time_pmr_jac = @elapsed begin
    M_pmr_jac, R_pmr_jac =
        my_pmr(A, Pr_jacobi, copy(M0), itmax, tol)
end

# PCG

time_pcg_id = @elapsed begin
    M_pcg_id, R_pcg_id =
        my_pcg(A, Pr_identity, copy(M0), itmax, tol)
end

time_pcg_jac = @elapsed begin
    M_pcg_jac, R_pcg_jac =
        my_pcg(A, Pr_jacobi, copy(M0), itmax, tol)
end

# LOPMR

time_lopmr_id = @elapsed begin
    M_lopmr_id, R_lopmr_id =
        my_lopmr(A, Pr_identity, copy(M0), itmax, tol)
end

time_lopmr_jac = @elapsed begin
    M_lopmr_jac, R_lopmr_jac =
        my_lopmr(A, Pr_jacobi, copy(M0), itmax, tol)
end


function print_summary(name, R_id, R_jac, time_id, time_jac)
    println("\n$name")
    println("-----------------------------------")
    println("Identity final residual: ", R_id[end])
    println("Jacobi final residual:   ", R_jac[end])
    println("Identity minimum:        ", minimum(R_id))
    println("Jacobi minimum:          ", minimum(R_jac))
    println("Identity runtime:        ", time_id)
    println("Jacobi runtime:          ", time_jac)
end

print_summary("PMR", R_pmr_id, R_pmr_jac, time_pmr_id, time_pmr_jac)
print_summary("PCG", R_pcg_id, R_pcg_jac, time_pcg_id, time_pcg_jac)
print_summary("LOPMR", R_lopmr_id, R_lopmr_jac, time_lopmr_id, time_lopmr_jac)


iteration_axis(R) = 0:(length(R) - 1)

p = plot(
    yscale = :log10,
    xlabel = "Iteration, i",
    ylabel = "Residual norm, ‖Rᵢ‖F",
    title = "$matrix_name — Identity versus Jacobi preconditioning",
    legend = :outerright,
    grid = true,
    size = (1200, 700),
    left_margin = 18 * Measures.mm,
    bottom_margin = 12 * Measures.mm,
    right_margin = 8 * Measures.mm
)

# PMR
plot!(p, iteration_axis(R_pmr_id), R_pmr_id,
      label = "PMR — Identity",
      color = :blue,
      linestyle = :dash,
      linewidth = 2)

plot!(p, iteration_axis(R_pmr_jac), R_pmr_jac,
      label = "PMR — Jacobi",
      color = :blue,
      linestyle = :solid,
      linewidth = 3)

# PCG
plot!(p, iteration_axis(R_pcg_id), R_pcg_id,
      label = "PCG — Identity",
      color = :red,
      linestyle = :dash,
      linewidth = 2)

plot!(p, iteration_axis(R_pcg_jac), R_pcg_jac,
      label = "PCG — Jacobi",
      color = :red,
      linestyle = :solid,
      linewidth = 3)

# LOPMR
plot!(p, iteration_axis(R_lopmr_id), R_lopmr_id,
      label = "LOPMR — Identity",
      color = :green,
      linestyle = :dash,
      linewidth = 2)

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
        "identity_vs_jacobi_$(matrix_name).png"
    )
)