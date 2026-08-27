using SparseArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using Serialization
using MatrixDepot

spai_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations")
full_dir = joinpath(@__DIR__, "..", "..", "SPAI Iterations with Fully Low Precision")
utils_dir = joinpath(@__DIR__, "..", "..", "Utils")

history_dir = joinpath(@__DIR__, "msc04515_full_lowprec_histories")
results_dir = joinpath(@__DIR__, "msc04515_full_lowprec_results")
matrix_dir = joinpath(@__DIR__, "msc04515_full_lowprec_matrices")
checkpoint_dir = joinpath(@__DIR__, "msc04515_nodrop_checkpoints")

mkpath(history_dir)
mkpath(results_dir)
mkpath(matrix_dir)
mkpath(checkpoint_dir)

include(joinpath(utils_dir, "frobenius.jl"))

include(joinpath(spai_dir, "my_pmr_spai.jl"))
include(joinpath(spai_dir, "my_pcg_spai.jl"))
include(joinpath(spai_dir, "my_lopmr_spai.jl"))
include(joinpath(spai_dir, "refine_spai_pattern.jl"))

include(joinpath(full_dir, "my_pmr_fullF32.jl"))
include(joinpath(full_dir, "my_pcg_fullF32.jl"))
include(joinpath(full_dir, "my_lopmr_fullF32.jl"))

function relative_residual(A,M)
    n = size(A,1)
    A64 = sparse(Float64.(A))
    M64 = sparse(Float64.(M))
    I64 = spdiagm(0 => ones(Float64,n))
    frob_norm(I64 - A64*M64) / sqrt(Float64(n))
end

function run_spai(name,label,f,A,Pr,M0,itmax,tol)
    println("\n$name $label")

    result = nothing
    t = @elapsed result = f(A,Pr,copy(M0),itmax,tol)

    M,h = result
    r = relative_residual(A,M)

    @printf("Iterations: %d\n",length(h)-1)
    @printf("Relative residual: %.6e\n",r)
    @printf("Construction time: %.4f s\n",t)

    return (
        M=M,
        history=h,
        residual=r,
        time=t,
        iterations=length(h)-1
    )
end

function refine_result(A,r)
    Mref = nothing
    t = @elapsed Mref = refine_spai_pattern(A,r.M)
    res = relative_residual(A,Mref)

    @printf("Refined residual: %.6e\n",res)
    @printf("Refinement time: %.4f s\n",t)

    return (M=Mref,residual=res,time=t)
end

function load_or_run(f,path,label)
    if isfile(path)
        println("\nCHECKPOINT FOUND — loading $label")
        return deserialize(path)
    end

    result = f()
    serialize(path,result)
    println("Checkpoint saved: ",path)

    return result
end

function save_result(name,r)
    serialize(joinpath(matrix_dir,"$(name).jls"),r.M)
    writedlm(joinpath(history_dir,"$(name)_history.txt"),r.history)
end

function save_refined(name,r)
    serialize(joinpath(matrix_dir,"$(name)_refined.jls"),r.M)
end

md = mdopen("Boeing/msc04515")
A = sparse(Float64.(md.A))
n = size(A,1)

itmax = 5000
tol = 1.0e-15

d64 = diag(A)
d32 = Float32.(d64)

any(iszero,d64) && error("A contains a zero diagonal entry.")

Pr64 = spdiagm(0 => 1.0 ./ d64)
Pr32 = spdiagm(0 => Float32(1) ./ d32)

M0_64 = spdiagm(0 => ones(Float64,n))
M0_32 = spdiagm(0 => ones(Float32,n))

println("Matrix: msc04515")
println("Size: ",size(A))
println("nnz(A): ",nnz(A))
println("itmax: ",itmax)
println("tol: ",tol)
println("dropping: NONE")

pmr64 = load_or_run(
    joinpath(checkpoint_dir,"pmr_float64_nodrop.jls"),
    "PMR Float64"
) do
    run_spai(
        "PMR","Float64",my_pmr,
        A,Pr64,M0_64,itmax,tol
    )
end
save_result("pmr_float64",pmr64)
GC.gc()

pcg64 = load_or_run(
    joinpath(checkpoint_dir,"pcg_float64_nodrop.jls"),
    "PCG Float64"
) do
    run_spai(
        "PCG","Float64",my_pcg,
        A,Pr64,M0_64,itmax,tol
    )
end
save_result("pcg_float64",pcg64)
GC.gc()

lopmr64 = load_or_run(
    joinpath(checkpoint_dir,"lopmr_float64_nodrop.jls"),
    "LOPMR Float64"
) do
    run_spai(
        "LOPMR","Float64",my_lopmr,
        A,Pr64,M0_64,itmax,tol
    )
end
save_result("lopmr_float64",lopmr64)
GC.gc()

pmr32 = load_or_run(
    joinpath(checkpoint_dir,"pmr_fullF32_nodrop.jls"),
    "PMR Full F32"
) do
    run_spai(
        "PMR","Fully Float32",my_pmr_fully_float32,
        A,Pr32,M0_32,itmax,tol
    )
end
save_result("pmr_fullF32",pmr32)
GC.gc()

pmr32r = load_or_run(
    joinpath(checkpoint_dir,"pmr_fullF32_refined_nodrop.jls"),
    "PMR Full F32 refinement"
) do
    println("\nPMR Full F32 + Float64 refinement")
    refine_result(A,pmr32)
end
save_refined("pmr_fullF32",pmr32r)
GC.gc()

pcg32 = load_or_run(
    joinpath(checkpoint_dir,"pcg_fullF32_nodrop.jls"),
    "PCG Full F32"
) do
    run_spai(
        "PCG","Fully Float32",my_pcg_fully_float32,
        A,Pr32,M0_32,itmax,tol
    )
end
save_result("pcg_fullF32",pcg32)
GC.gc()

pcg32r = load_or_run(
    joinpath(checkpoint_dir,"pcg_fullF32_refined_nodrop.jls"),
    "PCG Full F32 refinement"
) do
    println("\nPCG Full F32 + Float64 refinement")
    refine_result(A,pcg32)
end
save_refined("pcg_fullF32",pcg32r)
GC.gc()

lopmr32 = load_or_run(
    joinpath(checkpoint_dir,"lopmr_fullF32_nodrop.jls"),
    "LOPMR Full F32"
) do
    run_spai(
        "LOPMR","Fully Float32",my_lopmr_fully_float32,
        A,Pr32,M0_32,itmax,tol
    )
end
save_result("lopmr_fullF32",lopmr32)
GC.gc()

lopmr32r = load_or_run(
    joinpath(checkpoint_dir,"lopmr_fullF32_refined_nodrop.jls"),
    "LOPMR Full F32 refinement"
) do
    println("\nLOPMR Full F32 + Float64 refinement")
    refine_result(A,lopmr32)
end
save_refined("lopmr_fullF32",lopmr32r)
GC.gc()

println("\nFINAL RESIDUAL SUMMARY")

@printf(
    "%-8s %-14s %-14s %-14s\n",
    "Method","Float64","Full F32","F32 refined"
)

for (name,r64,r32,ref32) in (
    ("PMR",pmr64,pmr32,pmr32r),
    ("PCG",pcg64,pcg32,pcg32r),
    ("LOPMR",lopmr64,lopmr32,lopmr32r)
)
    @printf(
        "%-8s %-14.6e %-14.6e %-14.6e\n",
        name,r64.residual,r32.residual,ref32.residual
    )
end

println("\nITERATION SUMMARY")

@printf(
    "%-8s %-12s %-12s\n",
    "Method","Float64","Full F32"
)

for (name,r64,r32) in (
    ("PMR",pmr64,pmr32),
    ("PCG",pcg64,pcg32),
    ("LOPMR",lopmr64,lopmr32)
)
    @printf(
        "%-8s %-12d %-12d\n",
        name,r64.iterations,r32.iterations
    )
end

println("\nTIMING SUMMARY")

@printf(
    "%-8s %-12s %-12s %-12s %-12s\n",
    "Method","F64","F32","F32 ref","F32 total"
)

for (name,r64,r32,ref32) in (
    ("PMR",pmr64,pmr32,pmr32r),
    ("PCG",pcg64,pcg32,pcg32r),
    ("LOPMR",lopmr64,lopmr32,lopmr32r)
)
    @printf(
        "%-8s %-12.4f %-12.4f %-12.4f %-12.4f\n",
        name,
        r64.time,
        r32.time,
        ref32.time,
        r32.time+ref32.time
    )
end

summary = joinpath(
    results_dir,
    "msc04515_full_lowprec_summary.txt"
)

open(summary,"w") do io
    println(io,"msc04515 fully reduced Float32 + Float64 fixed-pattern refinement")
    println(io,"dropping: none")
    println(io,"itmax: ",itmax)
    println(io,"tol: ",tol)
    println(io)

    @printf(
        io,
        "%-8s %-14s %-14s %-14s %-10s %-10s\n",
        "Method","Float64","Full F32",
        "F32 refined","F64 it","F32 it"
    )

    for (name,r64,r32,ref32) in (
        ("PMR",pmr64,pmr32,pmr32r),
        ("PCG",pcg64,pcg32,pcg32r),
        ("LOPMR",lopmr64,lopmr32,lopmr32r)
    )
        @printf(
            io,
            "%-8s %-14.6e %-14.6e %-14.6e %-10d %-10d\n",
            name,
            r64.residual,
            r32.residual,
            ref32.residual,
            r64.iterations,
            r32.iterations
        )
    end

    println(io)

    @printf(
        io,
        "%-8s %-12s %-12s %-12s %-12s\n",
        "Method","F64","F32","F32 ref","F32 total"
    )

    for (name,r64,r32,ref32) in (
        ("PMR",pmr64,pmr32,pmr32r),
        ("PCG",pcg64,pcg32,pcg32r),
        ("LOPMR",lopmr64,lopmr32,lopmr32r)
    )
        @printf(
            io,
            "%-8s %-12.4f %-12.4f %-12.4f %-12.4f\n",
            name,
            r64.time,
            r32.time,
            ref32.time,
            r32.time+ref32.time
        )
    end
end

println("\nMSC04515 COMPLETE")
println("Checkpoints: ",checkpoint_dir)
println("Histories:   ",history_dir)
println("Matrices:    ",matrix_dir)
println("Summary:     ",summary)