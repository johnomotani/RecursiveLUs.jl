using Dates
using LinearAlgebra
using Profile
using StableRNGs
using StatProfilerHTML
using ColumnPivotLUs

function serial_profile(short_size::Integer, long_size::Integer, nsamples::Integer)
    BLAS.set_num_threads(1)

    if rank == 0
        println("""
                WARNING!!!
                ##########

                These profiles are essentially useless, because the sampling becomes very
                slow for large call stacks, which happen in the recursive_*_pivot_lu!()
                functions.  The cost of the sampling then skews the results so that the
                'blocked' and 'recursive' parts of the solve are not comparable.
                """)
    end

    rng = StableRNG(42)
    Ac = rand(short_size, long_size)
    Accopy = copy(Ac)
    jpiv = zeros(Int64, short_size)

    println("ColumnPivotLUMPI profile short_size=$short_size, long_size=$long_size, ", now())
    println("=====================================================================================")
    println()

    Acplu = get_column_pivot_lu(jpiv)

    lu!(Acplu, Accopy)

    @profile begin
        for i ∈ 1:nsamples
            Accopy .= Ac
            lu!(Acplu, Accopy)
        end
    end

    statprofilehtml(; path="statprof-ColumnPivotLU-$short_size-$long_size")

    Profile.clear()

    Ar = Matrix(transpose(Ac))
    Arcopy = copy(Ar)
    ipiv = zeros(Int64, short_size)

    println("RowPivotLUMPI profile long_size=$long_size, short_size=$short_size, ", now())
    println("=====================================================================================")
    println()

    Arplu = get_row_pivot_lu(ipiv)

    lu!(Arplu, Arcopy)

    @profile begin
        for i ∈ 1:nsamples
            Arcopy .= Ar
            lu!(Arplu, Arcopy)
        end
    end

    statprofilehtml(; path="statprof-RowPivotLU-$long_size-$short_size")

    return nothing
end

serial_profile(16, 16, 10000000)
serial_profile(64, 512, 100000)
#serial_profile(128, 4096, 40000)
#serial_profile(4096, 4096, 200)
#serial_profile(8192, 8192, 100)
