using BenchmarkTools
using Dates
using LinearAlgebra
using MPI
using Profile
using StableRNGs
using StatProfilerHTML
using ColumnPivotLUs

function mpi_profile(short_size::Integer, long_size::Integer, nsamples::Integer)
    BLAS.set_num_threads(1)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

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
    local_win_store_float = nothing
    local_win_store_float = MPI.Win[]
    allocate_shared_float = (dims...)->begin
        if rank == 0
            dims_local = dims
        else
            dims_local = Tuple(0 for _ ∈ dims)
        end
        win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local, comm)
        array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
        push!(local_win_store_float, win)
        if rank == 0
            array .= NaN
        end
        MPI.Barrier(comm)
        return array
    end

    local_win_store_int = MPI.Win[]
    allocate_shared_int = (dims...)->begin
        if rank == 0
            dims_local = dims
        else
            dims_local = Tuple(0 for _ ∈ dims)
        end
        win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local, comm)
        array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
        push!(local_win_store_int, win)
        if rank == 0
            array .= typemin(Int64)
        end
        MPI.Barrier(comm)
        return array
    end

    Accopy = allocate_shared_float(short_size, long_size)
    jpiv = allocate_shared_int(short_size)
    index_buffer = allocate_shared_int(nproc)
    maxabs_buffer = allocate_shared_float(nproc)
    if rank == 0
        rng = StableRNG(42)
        Ac = rand(short_size, long_size)

        Accopy .= Ac

        println("ColumnPivotLUMPI profile np=$nproc short_size=$short_size, long_size=$long_size, ", now())
        println("=====================================================================================")
        println()
    else
        Ac = nothing
    end
    Acplu = get_column_pivot_lu(jpiv, comm, index_buffer, maxabs_buffer)

    MPI.Barrier(comm)
    lu!(Acplu, Accopy)

    MPI.Barrier(comm)

    @profile begin
        for i ∈ 1:nsamples
            if rank == 0
                Accopy .= Ac
            end
            MPI.Barrier(comm)
            lu!(Acplu, Accopy)
            MPI.Barrier(comm)
        end
    end

    statprofilehtml(; path=joinpath("statprof-ColumnPivotLUMPI-$nproc", "profile-$rank"))

    Profile.clear()

    Arcopy = allocate_shared_float(long_size, short_size)
    ipiv = allocate_shared_int(short_size)
    index_buffer = allocate_shared_int(nproc)
    maxabs_buffer = allocate_shared_float(nproc)
    if rank == 0
        Ar = Matrix(transpose(Ac))

        Arcopy .= Ar

        println("RowPivotLUMPI profile np=$nproc long_size=$long_size, short_size=$short_size, ", now())
        println("=====================================================================================")
        println()
    else
        Ar = nothing
    end
    Arplu = get_row_pivot_lu(ipiv, comm, index_buffer, maxabs_buffer)

    MPI.Barrier(comm)
    lu!(Arplu, Arcopy)

    MPI.Barrier(comm)

    @profile begin
        for i ∈ 1:nsamples
            if rank == 0
                Arcopy .= Ar
            end
            MPI.Barrier(comm)
            lu!(Arplu, Arcopy)
            MPI.Barrier(comm)
        end
    end

    statprofilehtml(; path=joinpath("statprof-RowPivotLUMPI-$nproc", "profile-$rank"))

    # Free the MPI.Win objects, because if they are free'd by the garbage collector it may
    # cause an MPI error or hang.
    for w ∈ local_win_store_float
        MPI.free(w)
    end
    resize!(local_win_store_float, 0)
    for w ∈ local_win_store_int
        MPI.free(w)
    end
    resize!(local_win_store_int, 0)

    return nothing
end

MPI.Init()
#mpi_profile(128, 4096, 4000)
#mpi_profile(4096, 4096, 20)
mpi_profile(8192, 8192, 10)
