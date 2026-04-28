using MPISharedMemLUs
using LinearAlgebra
using MPI
using StableRNGs
using Test

function get_comms()
    comm = MPI.COMM_WORLD
    nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

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

    return comm, rank, nproc, allocate_shared_float, allocate_shared_int,
           local_win_store_float, local_win_store_int
end

function test_column_pivoting(m, n, tol)
    rng = StableRNG(234)
    @testset "column pivoting" begin
        A = rand(rng, m, n)
        s = min(m, n)
        jpiv = zeros(Int64, s)
        Alu = get_column_pivot_lu(jpiv)
        LU_arr = copy(A)
        lu!(Alu, LU_arr)
        L = fill(NaN, m, s)
        L[1:s,1:s] .= UnitLowerTriangular(LU_arr[1:s,1:s])
        L[s+1:end,1:s] .= LU_arr[s+1:end,1:s]
        U = fill(NaN, s, n)
        U[1:s,1:s] .= UpperTriangular(LU_arr[1:s,1:s])
        U[1:s,s+1:end] .= LU_arr[1:s,s+1:end]
        p = LinearAlgebra.ipiv2perm(Alu.jpiv, n)
        @test isapprox(L * U, A[:,p], atol=tol, norm=x->NaN)
    end
    return nothing
end

function test_column_pivoting_mpi(m, n, tol)
    rng = StableRNG(234)
    comm, rank, nproc, allocate_shared_float, allocate_shared_int, local_win_store_float,
        local_win_store_int = get_comms()

    @testset "column pivoting with mpi" begin
        LU_arr = allocate_shared_float(m, n)
        index_buffer = allocate_shared_int(nproc)
        maxabs_buffer = allocate_shared_float(nproc)
        if rank == 0
            A = rand(rng, m, n)
            LU_arr .= A
            index_buffer .= -1
            maxabs_buffer .= NaN
        end
        s = min(m, n)
        jpiv = allocate_shared_int(s)
        Alu = get_column_pivot_lu(jpiv, comm, index_buffer, maxabs_buffer)
        MPI.Barrier(comm)
        lu!(Alu, LU_arr)
        if rank == 0
            L = fill(NaN, m, s)
            L[1:s,1:s] .= UnitLowerTriangular(LU_arr[1:s,1:s])
            L[s+1:end,1:s] .= LU_arr[s+1:end,1:s]
            U = fill(NaN, s, n)
            U[1:s,1:s] .= UpperTriangular(LU_arr[1:s,1:s])
            U[1:s,s+1:end] .= LU_arr[1:s,s+1:end]
            p = LinearAlgebra.ipiv2perm(Alu.jpiv, n)
            @test isapprox(L * U, A[:,p], atol=tol, norm=x->NaN)
        end
    end

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

function test_row_pivoting(m, n, tol)
    rng = StableRNG(123)
    @testset "row pivoting" begin
        A = rand(rng, m, n)
        s = min(m, n)
        ipiv = zeros(Int64, s);
        Alu = get_row_pivot_lu(ipiv)
        LU_arr = copy(A)
        lu!(Alu, LU_arr)
        L = fill(NaN, m, s)
        L[1:s,1:s] .= UnitLowerTriangular(LU_arr[1:s,1:s])
        L[s+1:end,1:s] .= LU_arr[s+1:end,1:s]
        U = fill(NaN, s, n)
        U[1:s,1:s] .= UpperTriangular(LU_arr[1:s,1:s])
        U[1:s,s+1:end] .= LU_arr[1:s,s+1:end]
        p = LinearAlgebra.ipiv2perm(Alu.ipiv, m)
        @test isapprox(L * U, A[p,:], atol=tol, norm=x->NaN)
    end
    return nothing
end

function test_row_pivoting_mpi(m, n, tol)
    rng = StableRNG(123)
    comm, rank, nproc, allocate_shared_float, allocate_shared_int, local_win_store_float,
        local_win_store_int = get_comms()

    @testset "row pivoting with mpi" begin
        LU_arr = allocate_shared_float(m, n)
        index_buffer = allocate_shared_int(nproc)
        maxabs_buffer = allocate_shared_float(nproc)
        if rank == 0
            A = rand(rng, m, n)
            LU_arr .= A
            index_buffer .= -1
            maxabs_buffer .= NaN
        end
        s = min(m, n)
        ipiv = allocate_shared_int(s);
        Alu = get_row_pivot_lu(ipiv, comm)
        MPI.Barrier(comm)
        lu!(Alu, LU_arr)
        if rank == 0
            L = fill(NaN, m, s)
            L[1:s,1:s] .= UnitLowerTriangular(LU_arr[1:s,1:s])
            L[s+1:end,1:s] .= LU_arr[s+1:end,1:s]
            U = fill(NaN, s, n)
            U[1:s,1:s] .= UpperTriangular(LU_arr[1:s,1:s])
            U[1:s,s+1:end] .= LU_arr[1:s,s+1:end]
            p = LinearAlgebra.ipiv2perm(Alu.ipiv, m)
            @test isapprox(L * U, A[p,:], atol=tol, norm=x->NaN)
        end
    end

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

@testset "MPISharedMemLUs.jl" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    BLAS.set_num_threads(1)
    tol = 4.0e-13
    @testset "m=$m n=$n" for m ∈ [16, 32, 53, 64, 128, 143, 4096], n ∈ [16, 32, 53, 64, 128, 143, 4096]
        test_column_pivoting(m, n, tol)
        test_column_pivoting_mpi(m, n, tol)
        test_row_pivoting(m, n, tol)
        test_row_pivoting_mpi(m, n, tol)
    end
end
