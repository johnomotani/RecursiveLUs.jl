using BenchmarkTools
using LinearAlgebra
using ColumnPivotLUs

function serial_benchmark(short_size::Integer, long_size::Integer)
    # Here we compare serial version of all algorithms.
    BLAS.set_num_threads(1)

    dat_dir = "serial-benchmark-data"
    mkpath(dat_dir)

    Ac = rand(short_size,long_size)
    jpiv = zeros(Int64,short_size)
    Ar = Matrix(transpose(Ac))
    ipiv = zeros(Int64,short_size)

    println("Benchmark short_size=$short_size, long_size=$long_size")
    println("===========================================")
    println()

    println("LAPACK")
    b = @benchmark LAPACK.getrf!(Acopy, $ipiv; check=false) setup=(Acopy=copy($Ar))
    display(b)
    open(joinpath(dat_dir, "LAPACK.dat"), "a") do io
        println(io, minimum(b.times) * 1.0e-6)
    end

    println("\nRow pivoting")
    rplu = get_row_pivot_lu(ipiv)
    b = @benchmark lu!($rplu, Acopy) setup=(Acopy=copy($Ar))
    display(b)
    open(joinpath(dat_dir, "RP.dat"), "a") do io
        println(io, minimum(b.times) * 1.0e-6)
    end

    println("\nColumn pivoting")
    cplu = get_column_pivot_lu(jpiv)
    b = @benchmark lu!($cplu, Acopy) setup=(Acopy=copy($Ac))
    display(b)
    open(joinpath(dat_dir, "CP.dat"), "a") do io
        println(io, minimum(b.times) * 1.0e-6)
    end

    println()

    return nothing
end

serial_benchmark(1, 16)
serial_benchmark(2, 16)
serial_benchmark(4, 16)
serial_benchmark(8, 16)
serial_benchmark(16, 16)
serial_benchmark(64, 64)
serial_benchmark(64, 128)
serial_benchmark(64, 256)
serial_benchmark(64, 512)
serial_benchmark(64, 1024)
serial_benchmark(64, 2048)
serial_benchmark(1, 4096)
serial_benchmark(2, 4096)
serial_benchmark(4, 4096)
serial_benchmark(8, 4096)
serial_benchmark(8, 4096)
serial_benchmark(16, 4096)
serial_benchmark(32, 4096)
serial_benchmark(64, 4096)
serial_benchmark(128, 4096)
serial_benchmark(4096, 4096)
serial_benchmark(8192, 8192)
