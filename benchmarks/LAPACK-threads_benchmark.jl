using BenchmarkTools
using Dates
using LinearAlgebra
using StableRNGs
using ColumnPivotLUs

function threaded_benchmark(short_size::Integer, long_size::Integer)
    nt = Threads.nthreads()
    BLAS.set_num_threads(nt)

    rng = StableRNG(42)

    Ac = rand(rng, short_size, long_size)
    Ar = Matrix(transpose(Ac))
    ipiv = zeros(Int64,short_size)

    dat_dir = "benchmark-LAPACK-data"
    mkpath(dat_dir)

    println("LAPACK Benchmark nt=$nt short_size=$short_size, long_size=$long_size, ", now())
    println("============================================================================")
    println()

    b = @benchmark LAPACK.getrf!(Acopy, $ipiv; check=false) setup=(Acopy=copy($Ar))
    display(b)
    println()
    open(joinpath(dat_dir, "LAPACK-$long_size-$short_size.dat"), "a") do io
        println(io, minimum(b.times) * 1.0e-6)
    end

    return nothing
end

threaded_benchmark(128, 4096)
threaded_benchmark(4096, 4096)
threaded_benchmark(8192, 8192)
