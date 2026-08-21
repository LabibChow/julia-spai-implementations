using DelimitedFiles

history_dir = joinpath(
    @__DIR__,
    "tri100eigs4k_precision_histories",
)

pmr = vec(readdlm(
    joinpath(history_dir, "pmr_float64_history.txt")
))

pcg = vec(readdlm(
    joinpath(history_dir, "pcg_float64_history.txt")
))

lopmr = vec(readdlm(
    joinpath(history_dir, "lopmr_float64_history.txt")
))

println("PMR initial   = ", pmr[1])
println("PCG initial   = ", pcg[1])
println("LOPMR initial = ", lopmr[1])

println()

println("PMR length    = ", length(pmr))
println("PCG length    = ", length(pcg))
println("LOPMR length  = ", length(lopmr))