using SparseArrays
using LinearAlgebra
using Plots

include(joinpath(@__DIR__, "..", "..", "Utils", "check_spd.jl"))
include(joinpath(@__DIR__, "..", "..", "Utils", "frobenius.jl"))

include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_cg_solve.jl"))
include(joinpath(@__DIR__, "..", "..", "Linear Solvers", "my_pcg_solve.jl"))

include(joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision", "my_mr_float32_spai.jl"))

include(joinpath(@__DIR__, "..", "..", "SPAI Iterations with Lower Precision", "my_lomr_float32_spai.jl"))

# Mixed precision experiment
# SPAI construction: Float32
# SPD check: Float64
# PCG solver: Float64
# Matrix: tri100eigs4k

function read_mtx_coordinate(filename)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    open(filename, "r") do io
        header = readline(io)
        is_symmetric = occursin("symmetric", lowercase(header))

        line = readline(io)

        while startswith(line, "%")
            line = readline(io)
        end

        parts = split(strip(line))
        m = parse(Int, parts[1])
        n = parse(Int, parts[2])
        nnz_count = parse(Int, parts[3])

        for _ in 1:nnz_count
            parts = split(strip(readline(io)))

            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            v = parse(Float64, parts[3])

            push!(rows, i)
            push!(cols, j)
            push!(vals, v)

            if is_symmetric && i != j
                push!(rows, j)
                push!(cols, i)
                push!(vals, v)
            end
        end

        return sparse(rows, cols, vals, m, n)
    end
end

# 1. Load matrix

A64 = read_mtx_coordinate("Matrices/tri100eigs4k.mtx")
A64 = sparse(Float64.(A64))

n = size(A64, 1)

println("=====")
println("MIXED PRECISION EXPERIMENT")
println("Matrix: tri100eigs4k")
println("SPAI construction: Float32")
println("PCG solver: Float64")
println("Size: ", size(A64))
println("Nonzeros in A: ", nnz(A64))
println("=====")

# 2. Check original matrix

println("\nChecking whether original matrix A is SPD...")

spd_A = check_spd(A64; name = "tri100eigs4k")

if !spd_A.is_spd
    error("tri100eigs4k is not SPD, stopping experiment.")
end

# 3. Settings

spai_itmax = 40
spai_tol = 1e-3

solver_itmax = 1000
solver_tol = 1e-5

I64 = spdiagm(0 => ones(Float64, n))

# 4. Build MR-SPAI in Float32

println("\nBuilding Float32 MR-SPAI...")

M0_mr32 = spzeros(Float32, n, n)

M_mr32, res_mr32 =
    my_mr_float32(A64, M0_mr32, spai_itmax, spai_tol)

M_mr32_as64 =
    sparse(Float64.(M_mr32))

mr32_quality =
    norm(I64 - A64 * M_mr32_as64) / norm(I64)

println("\nChecking Float32 MR-SPAI after converting to Float64...")

spd_mr32 =
    check_spd(
        M_mr32_as64;
        name = "MR-SPAI Float32 evaluated in Float64",
        symmetry_tolerance = 1e-6
    )

# 5. Build LOMR-SPAI in Float32

println("\nBuilding Float32 LOMR-SPAI...")

M0_lomr32 = spzeros(Float32, n, n)

M_lomr32, res_lomr32 =
    my_lomr_float32(A64, M0_lomr32, spai_itmax, spai_tol)

M_lomr32_as64 =
    sparse(Float64.(M_lomr32))

lomr32_quality =
    norm(I64 - A64 * M_lomr32_as64) / norm(I64)

println("\nChecking Float32 LOMR-SPAI after converting to Float64...")

spd_lomr32 =
    check_spd(
        M_lomr32_as64;
        name = "LOMR-SPAI Float32 evaluated in Float64",
        symmetry_tolerance = 1e-6
    )

# 6. Create Ax = b

x_exact = ones(Float64, n)
b = A64 * x_exact
x0 = zeros(Float64, n)

# 7. Baseline Float64 CG

println("\nRunning Float64 CG baseline...")

x_cg, res_cg =
    my_cg_solve(A64, b, x0, solver_itmax, solver_tol)

cg_rel_res =
    norm(b - A64 * x_cg) / norm(b)

cg_rel_err =
    norm(x_cg - x_exact) / norm(x_exact)

# 8. Float64 PCG with Float32-built MR-SPAI

if spd_mr32.is_spd

    println("\nRunning Float64 PCG with Float32-built MR-SPAI...")

    x_pcg_mr32, res_pcg_mr32 =
        my_pcg_solve(A64, b, x0, M_mr32_as64, solver_itmax, solver_tol)

    pcg_mr32_rel_res =
        norm(b - A64 * x_pcg_mr32) / norm(b)

    pcg_mr32_rel_err =
        norm(x_pcg_mr32 - x_exact) / norm(x_exact)

else
    println("\nFloat32-built MR-SPAI is not SPD. PCG skipped.")

    res_pcg_mr32 = Float64[]
    pcg_mr32_rel_res = NaN
    pcg_mr32_rel_err = NaN
end

# 9. Float64 PCG with Float32-built LOMR-SPAI

if spd_lomr32.is_spd

    println("\nRunning Float64 PCG with Float32-built LOMR-SPAI...")

    x_pcg_lomr32, res_pcg_lomr32 =
        my_pcg_solve(A64, b, x0, M_lomr32_as64, solver_itmax, solver_tol)

    pcg_lomr32_rel_res =
        norm(b - A64 * x_pcg_lomr32) / norm(b)

    pcg_lomr32_rel_err =
        norm(x_pcg_lomr32 - x_exact) / norm(x_exact)

else
    println("\nFloat32-built LOMR-SPAI is not SPD. PCG skipped.")

    res_pcg_lomr32 = Float64[]
    pcg_lomr32_rel_res = NaN
    pcg_lomr32_rel_err = NaN
end

# 10. Plot SPAI construction convergence

p1 = plot(
    0:length(res_mr32)-1,
    Float64.(res_mr32),
    yscale = :log10,
    xlabel = "SPAI iteration",
    ylabel = "SPAI residual",
    title = "Float32 MR/LOMR-SPAI construction for tri100eigs4k",
    label = "MR-SPAI Float32",
    linewidth = 3
)

plot!(
    p1,
    0:length(res_lomr32)-1,
    Float64.(res_lomr32),
    label = "LOMR-SPAI Float32",
    linestyle = :dash,
    linewidth = 3
)

savefig(
    p1,
    "tri100eigs4k_float32_mr_lomr_spai_convergence.png"
)

# 11. Plot solver convergence

p2 = plot(
    0:length(res_cg)-1,
    res_cg,
    yscale = :log10,
    xlabel = "Solver iteration",
    ylabel = "Relative residual",
    title = "Float32-built SPAI with Float64 PCG for tri100eigs4k",
    label = "CG Float64 baseline",
    linewidth = 3
)

if spd_mr32.is_spd
    plot!(
        p2,
        0:length(res_pcg_mr32)-1,
        res_pcg_mr32,
        label = "PCG + MR-SPAI Float32",
        linestyle = :dash,
        linewidth = 3
    )
end

if spd_lomr32.is_spd
    plot!(
        p2,
        0:length(res_pcg_lomr32)-1,
        res_pcg_lomr32,
        label = "PCG + LOMR-SPAI Float32",
        linestyle = :dot,
        linewidth = 3
    )
end

savefig(
    p2,
    "tri100eigs4k_float32_spai_float64_pcg.png"
)

# 12. Summary

println("\n=====")
println("MIXED PRECISION SUMMARY")
println("=====")

println("\nBaseline Float64 CG")
println("Iterations: ", length(res_cg) - 1)
println("Final residual: ", cg_rel_res)
println("Relative error: ", cg_rel_err)

println("\nFloat32-built MR-SPAI")
println("SPAI quality evaluated in Float64: ", mr32_quality)
println("Symmetry error: ", spd_mr32.symmetry_error)
println("Smallest eigenvalue: ", spd_mr32.lambda_min)
println("Cholesky succeeded: ", spd_mr32.cholesky_success)
println("SPD: ", spd_mr32.is_spd)
println("Nonzeros in M: ", nnz(M_mr32))

if spd_mr32.is_spd
    println("PCG iterations: ", length(res_pcg_mr32) - 1)
    println("PCG final residual: ", pcg_mr32_rel_res)
    println("PCG relative error: ", pcg_mr32_rel_err)
else
    println("PCG skipped because MR-SPAI Float32 was not SPD.")
end

println("\nFloat32-built LOMR-SPAI")
println("SPAI quality evaluated in Float64: ", lomr32_quality)
println("Symmetry error: ", spd_lomr32.symmetry_error)
println("Smallest eigenvalue: ", spd_lomr32.lambda_min)
println("Cholesky succeeded: ", spd_lomr32.cholesky_success)
println("SPD: ", spd_lomr32.is_spd)
println("Nonzeros in M: ", nnz(M_lomr32))

if spd_lomr32.is_spd
    println("PCG iterations: ", length(res_pcg_lomr32) - 1)
    println("PCG final residual: ", pcg_lomr32_rel_res)
    println("PCG relative error: ", pcg_lomr32_rel_err)
else
    println("PCG skipped because LOMR-SPAI Float32 was not SPD.")
end

println("\nSaved plots:")
println("tri100eigs4k_float32_mr_lomr_spai_convergence.png")
println("tri100eigs4k_float32_spai_float64_pcg.png")