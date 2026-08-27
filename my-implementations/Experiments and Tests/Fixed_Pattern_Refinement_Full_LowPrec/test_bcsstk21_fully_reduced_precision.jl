using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using MatrixDepot

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")
full_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Fully Low Precision")
utils_dir = joinpath(@__DIR__, "..", "..", "Utils")

history_dir = joinpath(@__DIR__, "bcsstk21_full_lowprec_histories")
results_dir = joinpath(@__DIR__, "bcsstk21_full_lowprec_results")
mkpath(history_dir)
mkpath(results_dir)

include(joinpath(utils_dir, "frobenius.jl"))
include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))

include(joinpath(full_dir, "my_pmr_fullF32.jl"))
include(joinpath(full_dir, "my_pcg_fullF32.jl"))
include(joinpath(full_dir, "my_lopmr_fullF32.jl"))
include(joinpath(full_dir, "my_pmr_fullF16.jl"))
include(joinpath(full_dir, "my_pcg_fullF16.jl"))
include(joinpath(full_dir, "my_lopmr_fullF16.jl"))

function relative_residual(A, M)
    n = size(A,1)
    A64 = sparse(Float64.(A))
    M64 = sparse(Float64.(M))
    I64 = spdiagm(0 => ones(Float64,n))
    frob_norm(I64 - A64*M64) / sqrt(Float64(n))
end

function run_spai(name, precision, f, A, Pr, M0, itmax, tol)
    println("\n$name $precision")
    result = nothing
    t = @elapsed result = f(A,Pr,copy(M0),itmax,tol)
    M, hist = result
    r = relative_residual(A,M)

    @printf("Iterations: %d\n", length(hist)-1)
    @printf("Relative residual: %.6e\n", r)
    @printf("Construction time: %.4f s\n", t)

    return (M=M, history=hist, residual=r,
            time=t, iterations=length(hist)-1)
end

function refine_result(A, result)
    Mref = nothing
    t = @elapsed Mref = refine_spai_pattern(A,result.M)
    r = relative_residual(A,Mref)

    @printf("Refined residual: %.6e\n", r)
    @printf("Refinement time: %.4f s\n", t)

    return (M=Mref, residual=r, time=t)
end

data = mdopen("HB/bcsstk21")
A = sparse(Float64.(data.A))
n = size(A,1)

itmax = 1000
tol = 1.0e-15

d64 = diag(A)
any(iszero,d64) && error("Zero diagonal entry in A.")

d32 = Float32.(d64)
d16 = Float16.(d64)

Pr64 = spdiagm(0 => 1.0 ./ d64)
Pr32 = spdiagm(0 => Float32(1) ./ d32)
Pr16 = spdiagm(0 => Float16(1) ./ d16)

M0_64 = spdiagm(0 => ones(Float64,n))
M0_32 = spdiagm(0 => ones(Float32,n))
M0_16 = spdiagm(0 => ones(Float16,n))

println("Matrix: bcsstk21")
println("Size: ", size(A))
println("nnz(A): ", nnz(A))
println("itmax: ", itmax)
println("tol: ", tol)
println("dropping: NONE")

pmr64 = run_spai("PMR","Float64",my_pmr,A,Pr64,M0_64,itmax,tol)
GC.gc()
pcg64 = run_spai("PCG","Float64",my_pcg,A,Pr64,M0_64,itmax,tol)
GC.gc()
lopmr64 = run_spai("LOPMR","Float64",my_lopmr,A,Pr64,M0_64,itmax,tol)

GC.gc()
pmr32 = run_spai("PMR","fully Float32",my_pmr_fully_float32,A,Pr32,M0_32,itmax,tol)
GC.gc()
pcg32 = run_spai("PCG","fully Float32",my_pcg_fully_float32,A,Pr32,M0_32,itmax,tol)
GC.gc()
lopmr32 = run_spai("LOPMR","fully Float32",my_lopmr_fully_float32,A,Pr32,M0_32,itmax,tol)

println("\nF32 refinements")
pmr32_ref = refine_result(A,pmr32)
GC.gc()
pcg32_ref = refine_result(A,pcg32)
GC.gc()
lopmr32_ref = refine_result(A,lopmr32)

GC.gc()
pmr16 = run_spai("PMR","fully Float16",my_pmr_fully_float16,A,Pr16,M0_16,itmax,tol)
GC.gc()
pcg16 = run_spai("PCG","fully Float16",my_pcg_fully_float16,A,Pr16,M0_16,itmax,tol)
GC.gc()
lopmr16 = run_spai("LOPMR","fully Float16",my_lopmr_fully_float16,A,Pr16,M0_16,itmax,tol)

println("\nF16 refinements")
pmr16_ref = refine_result(A,pmr16)
GC.gc()
pcg16_ref = refine_result(A,pcg16)
GC.gc()
lopmr16_ref = refine_result(A,lopmr16)

println("\nFINAL RESIDUAL SUMMARY")
@printf("%-8s %-14s %-14s %-14s %-14s %-14s\n",
    "Method","Float64","Full F32","F32 refined","Full F16","F16 refined")

for (name,r64,r32,ref32,r16,ref16) in (
    ("PMR",pmr64,pmr32,pmr32_ref,pmr16,pmr16_ref),
    ("PCG",pcg64,pcg32,pcg32_ref,pcg16,pcg16_ref),
    ("LOPMR",lopmr64,lopmr32,lopmr32_ref,lopmr16,lopmr16_ref)
)
    @printf("%-8s %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e\n",
        name,r64.residual,r32.residual,ref32.residual,
        r16.residual,ref16.residual)
end

println("\nITERATION SUMMARY")
@printf("%-8s %-12s %-12s %-12s\n",
    "Method","Float64","Full F32","Full F16")

for (name,r64,r32,r16) in (
    ("PMR",pmr64,pmr32,pmr16),
    ("PCG",pcg64,pcg32,pcg16),
    ("LOPMR",lopmr64,lopmr32,lopmr16)
)
    @printf("%-8s %-12d %-12d %-12d\n",
        name,r64.iterations,r32.iterations,r16.iterations)
end

println("\nTIMING SUMMARY")
@printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
    "Method","F64","F32","F32 ref","F32 total",
    "F16","F16 ref","F16 total")

for (name,r64,r32,ref32,r16,ref16) in (
    ("PMR",pmr64,pmr32,pmr32_ref,pmr16,pmr16_ref),
    ("PCG",pcg64,pcg32,pcg32_ref,pcg16,pcg16_ref),
    ("LOPMR",lopmr64,lopmr32,lopmr32_ref,lopmr16,lopmr16_ref)
)
    @printf("%-8s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
        name,r64.time,r32.time,ref32.time,r32.time+ref32.time,
        r16.time,ref16.time,r16.time+ref16.time)
end

for (name,r64,r32,r16) in (
    ("pmr",pmr64,pmr32,pmr16),
    ("pcg",pcg64,pcg32,pcg16),
    ("lopmr",lopmr64,lopmr32,lopmr16)
)
    writedlm(joinpath(history_dir,"$(name)_float64_history.txt"),r64.history)
    writedlm(joinpath(history_dir,"$(name)_fullF32_history.txt"),r32.history)
    writedlm(joinpath(history_dir,"$(name)_fullF16_history.txt"),r16.history)
end

for (name,ref32,ref16) in (
    ("pmr",pmr32_ref,pmr16_ref),
    ("pcg",pcg32_ref,pcg16_ref),
    ("lopmr",lopmr32_ref,lopmr16_ref)
)
    writedlm(
        joinpath(history_dir,"$(name)_full_refined_residuals.txt"),
        [ref32.residual ref16.residual]
    )
end

summary_file = joinpath(results_dir,"bcsstk21_full_lowprec_summary.txt")

open(summary_file,"w") do io
    println(io,"bcsstk21 fully reduced precision + Float64 fixed-pattern refinement")
    println(io,"Dropping: none")
    println(io,"itmax: ",itmax)
    println(io,"tol: ",tol)
    println(io)

    @printf(io,"%-8s %-14s %-14s %-14s %-14s %-14s\n",
        "Method","Float64","Full F32","F32 refined","Full F16","F16 refined")

    for (name,r64,r32,ref32,r16,ref16) in (
        ("PMR",pmr64,pmr32,pmr32_ref,pmr16,pmr16_ref),
        ("PCG",pcg64,pcg32,pcg32_ref,pcg16,pcg16_ref),
        ("LOPMR",lopmr64,lopmr32,lopmr32_ref,lopmr16,lopmr16_ref)
    )
        @printf(io,"%-8s %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e\n",
            name,r64.residual,r32.residual,ref32.residual,
            r16.residual,ref16.residual)
    end
end

println("\nExperiment complete.")
println("Histories: ",history_dir)
println("Summary: ",summary_file)