"""
LU factorizations, using algorithms copied from LAPACK

Column-pivoting and row-pivoting variants of LU factorization. Parallelised using
shared-memory MPI.

The structs defined in ColumnPivotLUs are designed to be re-used with different sizes of
matrix, so do not include the matrix storage, as would be usual in `LinearAlgebra.LU`,
etc. The factors are returned in-place in the matrix passed to `lu!()`.

The parallel version assumes that the arrays passed in (apart from `jpiv`) are MPI
shared-memory arrays, which can be accessed by all processes in the MPI communicator.
"""
module ColumnPivotLUs

export get_column_pivot_lu, get_row_pivot_lu

#using LoopVectorization
using Combinatorics
using LinearAlgebra
using LinearAlgebra.BLAS: trsm!
using LinearAlgebra.LAPACK: getrf!
using MPI
using Primes
using TimerOutputs

import LinearAlgebra: lu!

# This is LAPACK's default block size for DGETRF()
const block_size = 64

# Total (m*n) matrix size to revert to serial solves for submatrices.
const serial_threshold = 0

struct ColumnPivotLU
    jpiv::Vector{Int64}
end

struct ColumnPivotLUMPI{Vecint,Vecfloat,Ttimer,Tsync}
    jpiv::Vector{Int64}
    comm::MPI.Comm
    index_buffer::Vecint
    maxabs_buffer::Vecfloat
    rank::Int64
    nproc::Int64
    proc_i::Int64
    proc_I::Int64
    proc_j::Int64
    proc_J::Int64
    use_rectangular_parallelism_threshold::Int64
    synchronize::Tsync
    timer::Ttimer
end

struct RowPivotLU
    ipiv::Vector{Int64}
end

struct RowPivotLUMPI{Vecint,Vecfloat,Ttimer,Tsync}
    ipiv::Vecint
    comm::MPI.Comm
    index_buffer::Vecint
    maxabs_buffer::Vecfloat
    rank::Int64
    nproc::Int64
    proc_i::Int64
    proc_I::Int64
    proc_j::Int64
    proc_J::Int64
    use_rectangular_parallelism_threshold::Int64
    synchronize::Tsync
    timer::Ttimer
end

macro maybe_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end

"""
    get_column_pivot_lu(jpiv::Vector{Int64})

`jpiv` does not have to be initialised, but must be longer than the largest number of
pivot elements (the smaller of total number of rows or columns for a matrix) in any matrix
that will be factorised using this `ColumnPivotLU`.
"""
function get_column_pivot_lu(jpiv::Vector{Int64})
    return ColumnPivotLU(jpiv)
end

"""
    get_column_pivot_lu(jpiv::Union{Vector{Int64},Nothing}, comm::MPI.Comm,
                        index_buffer::AbstractVector{<:Integer},
                        maxabs_buffer::AbstractVector{<:Number})

`jpiv` does not have to be initialised and is required only on the rank-0 process of
`comm`, but must be longer than the largest number of pivot elements (the smaller of total
number of rows or columns for a matrix) in any matrix that will be factorised using this
`ColumnPivotLU`.

`comm` is the MPI communicator containing processes used to parallelise factorizations
using this `ColumnPivotLUMPI`.

`index_buffer` and `maxabs_buffer` should be shared-memory arrays accessible by all
processes in `comm`, whose length is at least the size of `comm`.
"""
function get_column_pivot_lu(jpiv::Union{Vector{Int64},Nothing}, comm::MPI.Comm,
                             index_buffer::AbstractVector{<:Integer},
                             maxabs_buffer::AbstractVector{<:Number};
                             synchronize=nothing,
                             timer::Union{TimerOutput,Nothing}=nothing)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # Set up a rectangular grid of processes.
    nproc_factors =
        [prod(x) for x in collect(unique(combinations(factor(Vector, nproc))))]
    # Make rectangular grid as close to square as possible, with more columns than rows.
    # Find the last factor ≤ sqrt(nproc)
    factor_ind = findlast(x -> x≤sqrt(nproc), nproc_factors)
    proc_I = nproc_factors[factor_ind]
    proc_J = nproc ÷ proc_I
    proc_j, proc_i = divrem(rank, proc_I)

    # Approximate aspect ratio of processor grid is proc_J ÷ proc_I. When the matrix being
    # solved is a lot more rectangular than this, we want to switch to a linear process
    # grid.
    use_rectangular_parallelism_threshold = (8 * proc_I) ÷ proc_J

    if synchronize === nothing
        synchronize = () -> MPI.Barrier(comm)
    end

    return ColumnPivotLUMPI(jpiv, comm, index_buffer, maxabs_buffer, rank, nproc, proc_i,
                            proc_I, proc_j, proc_J, use_rectangular_parallelism_threshold,
                            synchronize, timer)
end

"""
    get_row_pivot_lu(ipiv::Vector{Int64})

`ipiv` does not have to be initialised, but must be longer than the largest number of
pivot elements (the smaller of total number of rows or columns for a matrix) in any matrix
that will be factorised using this `RowPivotLU`.
"""
function get_row_pivot_lu(ipiv::Vector{Int64})
    return RowPivotLU(ipiv)
end

"""
    get_row_pivot_lu(ipiv::AbstractVector{<:Integer}, comm::MPI.Comm,
                     index_buffer::AbstractVector{<:Integer},
                     maxabs_buffer::AbstractVector{<:Number})

`ipiv` does not have to be initialised and is a shared-memory array accessible on all
processes in `comm`, which must be longer than the largest number of pivot elements (the
smaller of total number of rows or columns for a matrix) in any matrix that will be
factorised using this `RowPivotLU`.

`comm` is the MPI communicator containing processes used to parallelise factorizations
using this `RowPivotLUMPI`.

`index_buffer` and `maxabs_buffer` should be shared-memory arrays accessible by all
processes in `comm`, whose length is at least the size of `comm`.
"""
function get_row_pivot_lu(ipiv::AbstractVector{<:Integer}, comm::MPI.Comm,
                          index_buffer::AbstractVector{<:Integer},
                          maxabs_buffer::AbstractVector{<:Number};
                          synchronize=nothing,
                          timer::Union{TimerOutput,Nothing}=nothing)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # Set up a rectangular grid of processes.
    nproc_factors =
        [prod(x) for x in collect(unique(combinations(factor(Vector, nproc))))]
    # Make rectangular grid as close to square as possible, with more rows than columns.
    # Find the last factor ≤ sqrt(nproc)
    factor_ind = findlast(x -> x≤sqrt(nproc), nproc_factors)
    proc_J = nproc_factors[factor_ind]
    proc_I = nproc ÷ proc_J
    proc_j, proc_i = divrem(rank, proc_I)

    # Approximate aspect ratio of processor grid is proc_I ÷ proc_J. When the matrix being
    # solved is a lot more rectangular than this, we want to switch to a linear process
    # grid.
    use_rectangular_parallelism_threshold = (8 * proc_J) ÷ proc_I

    if synchronize === nothing
        synchronize = () -> MPI.Barrier(comm)
    end

    return RowPivotLUMPI(ipiv, comm, index_buffer, maxabs_buffer, rank, nproc, proc_i,
                         proc_I, proc_j, proc_J, use_rectangular_parallelism_threshold,
                         synchronize, timer)
end

function find_pivot(a::AbstractVector, n::Integer)
    @inbounds begin
        pivot_ind = 1
        maxabs = abs(a[1])
        for j ∈ 2:n
            thisabs = abs(a[j])
            if thisabs > maxabs
                maxabs = thisabs
                pivot_ind = j
            end
        end
        return pivot_ind
    end
end

function find_pivot(xplu::Union{ColumnPivotLUMPI,RowPivotLUMPI}, a::AbstractVector,
                    n::Integer)
    @inbounds begin
        comm = xplu.comm
        index_buffer = xplu.index_buffer
        maxabs_buffer = xplu.maxabs_buffer
        rank = xplu.rank
        nproc = xplu.nproc
        synchronize = xplu.synchronize

        entries_per_proc = (n + nproc - 1) ÷ nproc
        first_local_entry = rank * entries_per_proc + 1
        last_local_entry = min((rank + 1) * entries_per_proc, n)
        if last_local_entry ≥ first_local_entry
            pivot_ind = first_local_entry
            maxabs = abs(a[first_local_entry])
            for j ∈ first_local_entry+1:last_local_entry
                thisabs = abs(a[j])
                if thisabs > maxabs
                    maxabs = thisabs
                    pivot_ind = j
                end
            end
            index_buffer[rank+1] = pivot_ind
            maxabs_buffer[rank+1] = maxabs
        else
            index_buffer[rank+1] = -1
            maxabs_buffer[rank+1] = -1.0
        end
        synchronize()
        if rank == 0
            i = argmax(@view(maxabs_buffer[1:nproc]))
            pivot_ind = index_buffer[i]
        else
            pivot_ind = -1
        end
        return pivot_ind
    end
end

function apply_column_swaps!(A, jpiv, m, npivot)
    @inbounds begin
        for j ∈ 1:npivot
            pivot_ind = jpiv[j]
            for i ∈ 1:m
                A[i,j], A[i,pivot_ind] = A[i,pivot_ind], A[i,j]
            end
        end
    end
    return nothing
end

function lu!(cplu::ColumnPivotLU, A::AbstractMatrix)
    blocked_column_pivot_lu!(cplu.jpiv, A, size(A, 1), size(A, 2))
    return A
end

function blocked_column_pivot_lu!(jpiv::AbstractVector{<:Integer}, A::AbstractMatrix,
                                  m::Integer, n::Integer)
    @inbounds begin
        n_diag = min(m, n)

        if n_diag ≤ block_size
            return recursive_column_pivot_lu!(jpiv, A, m, n)
        end

        for i ∈ 1:block_size:n_diag
            ib = min(block_size, n_diag - i + 1)
            ie = i + ib - 1
            this_jpiv = @view jpiv[i:n_diag]

            # Factor diagonal and right-of-diagonal blocks.
            @views recursive_column_pivot_lu!(this_jpiv, A[i:ie,i:n], ib, n - i + 1)

            # Apply interchanges to rows 1:i-1.
            if i > 1
                apply_column_swaps!(@view(A[1:i-1,i:n]), this_jpiv, i - 1, ib)
            end

            if i + ib ≤ m
                m2 = m - ie
                n2 = n - ie

                # Apply interchanges to rows i+ib:m.
                apply_column_swaps!(@view(A[ie+1:m,i:n]), this_jpiv, m2, ib)

                # Compute block column of L.
                A21 = @view A[ie+1:m,i:ie]
                @views trsm!('R', 'U', 'N', 'N', 1.0, A[i:ie,i:ie], A21)
                #A11 = @view A[i:ie,i:ie]
                #for i ∈ 1:m2, j ∈ 1:jb
                #    for k ∈ 1:j-1
                #        A21[i,j] -= A11[k,j] * A21[i,k]
                #    end
                #    A21[i,j] /= A11[j,j]
                #end

                if i + ib ≤ n
                    # Update trailing submatrix.
                    A12 = @view A[i:ie,ie+1:n]
                    A22 = @view A[ie+1:m,ie+1:n]
                    mul!(A22, A21, A12, -1.0, 1.0)
                    #@turbo for j ∈ 1:n2, k ∈ 1:jb, i ∈ 1:m2
                    #    A22[i,j] -= A21[i,k] * A12[k,j]
                    #end
                end
            end

            # Adjust pivot indices.
            this_jpiv .+= i - 1
        end
    end

    return nothing
end

function recursive_column_pivot_lu!(jpiv::AbstractVector{<:Integer}, A::AbstractMatrix,
                                    m::Integer, n::Integer)
    # A - the matrix being factorised in-place.
    # jpiv - the (column) pivot indices.
    # m - the number of rows in A.
    # n - the number of columns in A.

    # This function borrows heavily from DGETRF2 from LAPACK, v3.12.1.
    # Recurse not over rows/columns but by splitting the matrix approximately in half each
    # step.

    @inbounds begin
        # Quick return if possible.
        if m == 0 || n == 0
            return nothing
        end

        if n == 1
            # One column case, just need to handle jpiv and update column.
            jpiv[1] = 1
            @views A[2:end,1] .*= 1.0 / A[1,1]
        elseif m == 1
            # One row case.
            pivot_ind = find_pivot(@view(A[1,:]), n)
            jpiv[1] = pivot_ind

            # Apply the interchange
            A[1,1], A[1,pivot_ind] = A[1,pivot_ind], A[1,1]
        else
            n_diag = min(m, n)

            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            m1 = n_diag ÷ 2
            m2 = m - m1
            n2 = n - m1

            # Factor
            # [ A11 | A12 ]
            recursive_column_pivot_lu!(jpiv, @view(A[1:m1,:]), m1, n)

            # Apply interchanges to
            # [ A21 | A22 ]
            apply_column_swaps!(@view(A[m1+1:m,:]), jpiv, m2, m1)

            # Solve A21
            A21 = @view A[m1+1:m,1:m1]
            @views trsm!('R', 'U', 'N', 'N', 1.0, A[1:m1,1:m1], A21)
            #A11 = @view A[1:m1,1:m1]
            #for i ∈ 1:m2, j ∈ 1:m1
            #    for k ∈ 1:j-1
            #        A21[i,j] -= A11[k,j] * A21[i,k]
            #    end
            #    A21[i,j] /= A11[j,j]
            #end

            # Update A22
            A12 = @view A[1:m1,m1+1:n]
            A22 = @view A[m1+1:m,m1+1:n]
            mul!(A22, A21, A12, -1.0, 1.0)
            #@turbo for j ∈ 1:n2, k ∈ 1:m1, i ∈ 1:m2
            #    A22[i,j] -= A21[i,k] * A12[k,j]
            #end

            # Factor A22
            right_jpiv = @view jpiv[m1+1:n_diag]
            recursive_column_pivot_lu!(right_jpiv, A22, m2, n2)

            # Apply interchanges to A12.
            apply_column_swaps!(A12, right_jpiv, m1, min(m2,n2))

            right_jpiv .+= m1
        end
    end
    return nothing
end

# MPI Parallel versions use shared-memory parallelism, so assume that `A` is an MPI
# shared-memory array. `index_buffer` and `maxabs_buffer should also be shared-memory
# arrays. `jpiv` does not have to be shared-memory and is only used/filled on rank-0.
# Assuming that `column_pivot_lu!()` is most often called on short, wide matrices (because
# it is used to factorise 'top panels' as part of a blocked algorithm), we divide columns
# among processes, and do not divide rows.

function lu!(cplu::ColumnPivotLUMPI, A::AbstractMatrix)
    @maybe_timeit cplu.timer "blocked_column_pivot_lu!" begin
        blocked_column_pivot_lu!(cplu, A, size(A, 1), size(A, 2))
    end
    return A
end

function blocked_column_pivot_lu!(cplu::ColumnPivotLUMPI, A::AbstractMatrix, m::Integer,
                                  n::Integer)
    @inbounds begin
        jpiv = cplu.jpiv
        comm = cplu.comm
        rank = cplu.rank
        nproc = cplu.nproc
        proc_i = cplu.proc_i
        proc_j = cplu.proc_j
        proc_I = cplu.proc_I
        proc_J = cplu.proc_J
        rectangular_threshold = cplu.use_rectangular_parallelism_threshold
        synchronize = cplu.synchronize
        n_diag = min(m, n)

        if n_diag ≤ block_size
            return recursive_column_pivot_lu!(cplu, A, jpiv, m, n)
        elseif m * n < serial_threshold
            # For small (sub-)matrices, revert to a serial solve.
            if rank == 0
                return blocked_column_pivot_lu!(jpiv, A, m, n)
            else
                return nothing
            end
        end

        rectangular_parallelism = true
        for i ∈ 1:block_size:n_diag
            @maybe_timeit cplu.timer "i=$i" begin
                ib = min(block_size, n_diag - i + 1)
                ie = i + ib - 1
                if rank == 0
                    this_jpiv = @view jpiv[i:n_diag]
                else
                    # Not used, so just pass through jpiv.
                    this_jpiv = @view jpiv[1:0]
                end

                # Factor diagonal and right-of-diagonal blocks.
                @maybe_timeit cplu.timer "recursive_column_pivot_lu!" begin
                    @views recursive_column_pivot_lu!(cplu, A[i:ie,i:n], this_jpiv, ib, n - i + 1)
                end

                # Apply interchanges to rows 1:i-1.
                # Column swaps are not parallelised, because memory copies probably do not
                # benefit much from parallism (limited just by memory bandwidth) and the
                # swapping is inherently sequential.
                if rank == 0 && i > 1
                    @maybe_timeit cplu.timer "apply_column_swaps! upper" begin
                        apply_column_swaps!(@view(A[1:i-1,i:n]), this_jpiv, i - 1, ib)
                    end
                end

                if i + ib ≤ m
                    m2 = m - ie
                    n2 = n - ie

                    # Once we switch off rectangular parallelism, the aspect ratio
                    # (width/height) of the matrix only gets larger, so we can stop checking.
                    if rectangular_parallelism && n2 > m2 * rectangular_threshold
                        rectangular_parallelism = false
                    end

                    # Apply interchanges to rows i+ib:m.
                    # Column swaps are not parallelised, because memory copies probably do not
                    # benefit much from parallism (limited just by memory bandwidth) and the
                    # swapping is inherently sequential.
                    if rank == 0
                        @maybe_timeit cplu.timer "apply_column_swaps!" begin
                            apply_column_swaps!(@view(A[ie+1:m,i:n]), this_jpiv, m2, ib)
                        end
                    end

                    @maybe_timeit cplu.timer "synchronize 1" begin
                        synchronize()
                    end

                    # Compute block column of L.
                    @maybe_timeit cplu.timer "trsm!" begin
                        rows_per_proc = (m2 + nproc - 1) ÷ nproc
                        row_range = rank*rows_per_proc+ie+1:min((rank+1)*rows_per_proc+ie,m)
                        if !isempty(row_range)
                            A21 = @view A[row_range,i:ie]
                            @views trsm!('R', 'U', 'N', 'N', 1.0, A[i:ie,i:ie], A21)
                        end
                    end

                    @maybe_timeit cplu.timer "synchronize 2" begin
                        synchronize()
                    end

                    if i + ib ≤ n
                        @maybe_timeit cplu.timer "mul!" begin
                            # Update trailing submatrix.
                            if rectangular_parallelism
                                cols_per_proc = (n2 + proc_J - 1) ÷ proc_J
                                col_range = proc_j*cols_per_proc+ie+1:min((proc_j+1)*cols_per_proc+ie,n)
                                rows_per_proc = (m2 + proc_I - 1) ÷ proc_I
                                row_range = proc_i*rows_per_proc+ie+1:min((proc_i+1)*rows_per_proc+ie,m)
                            else
                                cols_per_proc = (n2 + nproc - 1) ÷ nproc
                                col_range = rank*cols_per_proc+ie+1:min((rank+1)*cols_per_proc+ie,n)
                                row_range = ie+1:m
                            end
                            if !isempty(col_range) && !isempty(row_range)
                                A21 = @view A[row_range,i:ie]
                                A12 = @view A[i:ie,col_range]
                                A22 = @view A[row_range,col_range]
                                mul!(A22, A21, A12, -1.0, 1.0)
                            end
                        end
                    end

                    @maybe_timeit cplu.timer "synchronize 3" begin
                        synchronize()
                    end
                end

                # Adjust pivot indices.
                this_jpiv .+= i - 1
            end
        end
    end

    return nothing
end

function recursive_column_pivot_lu!(cplu::ColumnPivotLUMPI, A::AbstractMatrix,
                                    jpiv::AbstractVector{<:Integer}, m::Integer,
                                    n::Integer)
    # A - the matrix being factorised in-place.
    # jpiv - the (column) pivot indices.
    # m - the number of rows in A.
    # n - the number of columns in A.
    # comm - MPI communicator linking the shared-memory processes.
    # index_buffer - a shared-memory integer buffer to use when finding pivot indices.
    # rank - the rank of this process in `comm`.
    # nproc - the number of processes in `comm`.

    # This function borrows heavily from DGETRF2 from LAPACK, v3.12.1.
    # Recurse not over rows/columns but by splitting the matrix approximately in half each
    # step.

    @inbounds begin
        # Quick return if possible.
        if m == 0 || n == 0
            return nothing
        end

        comm = cplu.comm
        rank = cplu.rank
        nproc = cplu.nproc
        synchronize = cplu.synchronize

        if m * n < serial_threshold
            # For small (sub-)matrices, revert to a serial solve.
            if rank == 0
                return recursive_column_pivot_lu!(jpiv, A, m, n)
            else
                return nothing
            end
        end

        if n == 1
            # One column case, just need to handle jpiv and update column.
            if rank == 0
                jpiv[1] = 1
            end
            rows_per_proc = (m - 1 + nproc - 1) ÷ nproc
            row_range = rank*rows_per_proc+2:min((rank+1)*rows_per_proc+1,m)
            @views A[row_range,1] .*= 1.0 / A[1,1]
        elseif m == 1
            # One row case.
            pivot_ind = find_pivot(cplu, @view(A[1,:]), n)
            if rank == 0
                jpiv[1] = pivot_ind

                # Apply the interchange
                A[1,1], A[1,pivot_ind] = A[1,pivot_ind], A[1,1]
            end
        else
            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            m1 = min(m, n) ÷ 2
            m2 = m - m1
            n2 = n - m1

            # Factor
            # [ A11 | A12 ]
            recursive_column_pivot_lu!(cplu, @view(A[1:m1,:]), jpiv, m1, n)

            # Apply interchanges to
            # [ A21 | A22 ]
            # Column swaps are not parallelised, because memory copies probably do not
            # benefit much from parallism (limited just by memory bandwidth) and the
            # swapping is inherently sequential.
            if rank == 0
                apply_column_swaps!(@view(A[m1+1:m,:]), jpiv, m2, m1)
            end

            synchronize()

            # Solve A21
            rows_per_proc = (m2 + nproc - 1) ÷ nproc
            row_range = rank*rows_per_proc+m1+1:min((rank+1)*rows_per_proc+m1,m)
            if !isempty(row_range)
                A21 = @view A[row_range,1:m1]
                @views trsm!('R', 'U', 'N', 'N', 1.0, A[1:m1,1:m1], A21)
            end

            synchronize()

            # Update A22
            cols_per_proc = (n2 + nproc - 1) ÷ nproc
            col_range = rank*cols_per_proc+m1+1:min((rank+1)*cols_per_proc+m1,n)
            if !isempty(col_range)
                A21 = @view A[m1+1:m,1:m1]
                A12 = @view A[1:m1,col_range]
                A22 = @view A[m1+1:m,col_range]
                mul!(A22, A21, A12, -1.0, 1.0)
            end

            synchronize()

            # Factor A22
            if rank == 0
                right_jpiv = @view jpiv[m1+1:min(m,n)]
            else
                # Not used, so just pass through jpiv.
                right_jpiv = jpiv
            end
            recursive_column_pivot_lu!(cplu, @view(A[m1+1:m,m1+1:n]), right_jpiv, m2, n2)

            # Apply interchanges to A12.
            if rank == 0
                apply_column_swaps!(@view(A[1:m1,m1+1:n]), right_jpiv, m1, min(m2,n2))

                right_jpiv .+= m1
            end
        end
        synchronize()
    end
    return nothing
end

function apply_row_swaps!(A, ipiv, n, mpivot)
    # Algorithm copied from LAPACK's DLASWP()
    @inbounds begin
        n32 = (n ÷ 32) * 32
        if n32 != 0
            for j ∈ 1:32:n32
                for i ∈ 1:mpivot
                    pivot = ipiv[i]
                    if pivot != i
                        for k ∈ j:j+31
                            A[i,k], A[pivot,k] = A[pivot,k], A[i,k]
                        end
                    end
                end
            end
        end
        if n32 != n
            j = n32 + 1
            for i ∈ 1:mpivot
                pivot = ipiv[i]
                if pivot != i
                    for k ∈ j:n
                        A[i,k], A[pivot,k] = A[pivot,k], A[i,k]
                    end
                end
            end
        end
    end
    return nothing
end

function lu!(rplu::RowPivotLU, A::AbstractMatrix)
    blocked_row_pivot_lu!(rplu.ipiv, A, size(A, 1), size(A, 2))
    return A
end

function blocked_row_pivot_lu!(ipiv::AbstractVector{<:Integer}, A::AbstractMatrix,
                               m::Integer, n::Integer)
    @inbounds begin
        n_diag = min(m, n)

        if n_diag ≤ block_size
            return recursive_row_pivot_lu!(ipiv, A, m, n)
        end

        for j ∈ 1:block_size:n_diag
            jb = min(block_size, n_diag - j + 1)
            je = j + jb - 1
            this_ipiv = @view ipiv[j:n_diag]

            # Factor diagonal and subdiagonal blocks.
            @views recursive_row_pivot_lu!(this_ipiv, A[j:m,j:je], m - j + 1, jb)

            # Apply interchanges to columns 1:j-1.
            if j > 1
                apply_row_swaps!(@view(A[j:m,1:j-1]), this_ipiv, j - 1, jb)
            end

            if j + jb ≤ n
                m2 = m - je
                n2 = n - je

                # Apply interchanges to columns j+jb:n.
                apply_row_swaps!(@view(A[j:m,je+1:n]), this_ipiv, n2, jb)

                # Compute block row of U.
                A12 = @view A[j:je,je+1:n]
                @views trsm!('L', 'L', 'N', 'U', 1.0, A[j:je,j:je], A12)
                #A11 = @view A[j:je,j:je]
                #for j ∈ 1:n2, i ∈ 1:jb-1
                #    for k ∈ i+1:jb
                #        A12[k,j] -= A11[k,i] * A12[i,j]
                #    end
                #end

                if j + jb ≤ m
                    # Update trailing submatrix.
                    A21 = @view A[je+1:m,j:je]
                    A22 = @view A[je+1:m,je+1:n]
                    mul!(A22, A21, A12, -1.0, 1.0)
                    #@turbo for j ∈ 1:n2, k ∈ 1:jb, i ∈ 1:m2
                    #    A22[i,j] -= A21[i,k] * A12[k,j]
                    #end
                end
            end

            # Adjust pivot indices.
            this_ipiv .+= j - 1
        end
    end

    return nothing
end

function recursive_row_pivot_lu!(ipiv::AbstractVector{<:Integer}, A::AbstractMatrix,
                                 m::Integer, n::Integer)
    # A - the matrix being factorised in-place.
    # ipiv - the (row) pivot indices.
    # m - the number of rows in A.
    # n - the number of columns in A.

    # This function is essentially a copy of DGETRF2 from LAPACK, v3.12.1.
    # Recurse not over rows/columns but by splitting the matrix approximately in half each
    # step.

    @inbounds begin
        # Quick return if possible.
        if m == 0 || n == 0
            return nothing
        end

        if m == 1
            # One row case, just need to handle ipiv.
            ipiv[1] = 1
        elseif n == 1
            # One column case.
            pivot_ind = find_pivot(@view(A[:,1]), m)
            ipiv[1] = pivot_ind

            # Apply the interchange
            A[1,1], A[pivot_ind,1] = A[pivot_ind,1], A[1,1]

            # Update the column
            @views A[2:end,1] .*= 1.0 / A[1,1]
        else
            n_diag = min(m, n)

            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            n1 = n_diag ÷ 2
            n2 = n - n1
            m2 = m - n1

            # Factor
            # [ A11 ]
            # [ --- ]
            # [ A21 ]
            recursive_row_pivot_lu!(ipiv, @view(A[:,1:n1]), m, n1)

            # Apply interchanges to
            # [ A12 ]
            # [ --- ]
            # [ A22 ]
            apply_row_swaps!(@view(A[:,n1+1:n]), ipiv, n2, n1)

            # Solve A12
            A12 = @view A[1:n1,n1+1:n]
            @views trsm!('L', 'L', 'N', 'U', 1.0, A[1:n1,1:n1], A12)
            #A11 = @view A[1:n1,1:n1]
            #for j ∈ 1:n2, i ∈ 1:n1-1
            #    for k ∈ i+1:n1
            #        A12[k,j] -= A11[k,i] * A12[i,j]
            #    end
            #end

            # Update A22
            A21 = @view A[n1+1:m,1:n1]
            A22 = @view A[n1+1:m,n1+1:n]
            mul!(A22, A21, A12, -1.0, 1.0)
            #@turbo for j ∈ 1:n2, k ∈ 1:n1, i ∈ 1:m2
            #    A22[i,j] -= A21[i,k] * A12[k,j]
            #end

            # Factor A22
            bottom_ipiv = @view ipiv[n1+1:n_diag]
            recursive_row_pivot_lu!(bottom_ipiv, A22, m2, n2)

            # Apply interchanges to A21.
            apply_row_swaps!(A21, bottom_ipiv, n1, min(m2,n2))

            bottom_ipiv .+= n1
        end
    end
    return nothing
end

function lu!(rplu::RowPivotLUMPI, A::AbstractMatrix)
    @maybe_timeit rplu.timer "blocked_row_pivot_lu!" begin
        blocked_row_pivot_lu!(rplu, A, size(A, 1), size(A, 2))
    end
    return A
end

function blocked_row_pivot_lu!(rplu::RowPivotLUMPI, A::AbstractMatrix, m::Integer,
                               n::Integer)
    @inbounds begin
        ipiv = rplu.ipiv
        comm = rplu.comm
        rank = rplu.rank
        nproc = rplu.nproc
        proc_i = rplu.proc_i
        proc_j = rplu.proc_j
        proc_I = rplu.proc_I
        proc_J = rplu.proc_J
        rectangular_threshold = rplu.use_rectangular_parallelism_threshold
        synchronize = rplu.synchronize
        n_diag = min(m, n)

        if n_diag ≤ block_size
            return recursive_row_pivot_lu!(rplu, A, ipiv, m, n)
        elseif m * n < serial_threshold
            # For small (sub-)matrices, revert to a serial solve.
            if rank == 0
                #return blocked_row_pivot_lu!(ipiv, A, m, n)
                return getrf!(A, ipiv; check=false)
            else
                return nothing
            end
        end

        rectangular_parallelism = true
        for j ∈ 1:block_size:n_diag
            @maybe_timeit rplu.timer "j=$j" begin
                jb = min(block_size, n_diag - j + 1)
                je = j + jb - 1
                this_ipiv = @view ipiv[j:n_diag]

                # Factor diagonal and subdiagonal blocks.
                @maybe_timeit rplu.timer "left panel factorisation" begin
                    #@views recursive_row_pivot_lu!(rplu, A[j:m,j:je], this_ipiv, m - j + 1, jb)
                    if rank == 0
                        #@views recursive_row_pivot_lu!(this_ipiv, A[j:m,j:je], m - j + 1, jb)
                        @views getrf!(A[j:m,j:je], this_ipiv; check=false)
                    end
                end
                @maybe_timeit rplu.timer "synchronize 1" begin
                    synchronize()
                end

                # Apply interchanges to columns 1:j-1.
                if j > 1
                    @maybe_timeit rplu.timer "apply_row_swaps! left" begin
                        cols_per_proc = (j - 1 + nproc - 1) ÷ nproc
                        col_range = rank*cols_per_proc+1:min((rank+1)*cols_per_proc,j-1)
                        if !isempty(col_range)
                            apply_row_swaps!(@view(A[j:m,col_range]), this_ipiv,
                                             length(col_range), jb)
                        end
                    end
                end

                if j + jb ≤ n
                    m2 = m - je
                    n2 = n - je

                    # Once we switch off rectangular parallelism, the aspect ratio
                    # (width/height) of the matrix only gets smaller, so we can stop checking.
                    if rectangular_parallelism && m2 > n2 * rectangular_threshold
                        rectangular_parallelism = false
                    end

                    cols_per_proc = (n2 + nproc - 1) ÷ nproc
                    col_range = rank*cols_per_proc+je+1:min((rank+1)*cols_per_proc+je,n)

                    if !isempty(col_range)
                        # Apply interchanges to columns j+jb:n.
                        @maybe_timeit rplu.timer "apply_row_swaps!" begin
                            apply_row_swaps!(@view(A[j:m,col_range]), this_ipiv,
                                             length(col_range), jb)
                        end

                        # Compute block row of U.
                        @maybe_timeit rplu.timer "trsm!" begin
                            A12 = @view A[j:je,col_range]
                            @views trsm!('L', 'L', 'N', 'U', 1.0, A[j:je,j:je], A12)
                        end
                    end

                    @maybe_timeit rplu.timer "synchronize 2" begin
                        synchronize()
                    end

                    if j + jb ≤ m
                        @maybe_timeit rplu.timer "mul!" begin
                            # Update trailing submatrix.
                            if rectangular_parallelism
                                cols_per_proc = (n2 + proc_J - 1) ÷ proc_J
                                col_range = proc_j*cols_per_proc+je+1:min((proc_j+1)*cols_per_proc+je,n)
                                rows_per_proc = (m2 + proc_I - 1) ÷ proc_I
                                row_range = proc_i*rows_per_proc+je+1:min((proc_i+1)*rows_per_proc+je,m)
                            else
                                col_range = je+1:n
                                rows_per_proc = (m2 + nproc - 1) ÷ nproc
                                row_range = rank*rows_per_proc+je+1:min((rank+1)*rows_per_proc+je,m)
                            end
                            if !isempty(col_range) && !isempty(row_range)
                                A21 = @view A[row_range,j:je]
                                A12 = @view A[j:je,col_range]
                                A22 = @view A[row_range,col_range]
                                mul!(A22, A21, A12, -1.0, 1.0)
                            end
                        end
                    end
                else
                    @maybe_timeit rplu.timer "synchronize 2" begin
                        synchronize()
                    end
                end

                # Adjust pivot indices.
                if rank == 0
                    this_ipiv .+= j - 1
                end
                @maybe_timeit rplu.timer "synchronize 3" begin
                    synchronize()
                end
            end
        end
    end

    return nothing
end

function recursive_row_pivot_lu!(rplu::RowPivotLUMPI, A::AbstractMatrix,
                                 ipiv::AbstractVector{<:Integer}, m::Integer, n::Integer)
    # A - the matrix being factorised in-place.
    # ipiv - the (row) pivot indices.
    # m - the number of rows in A.
    # n - the number of columns in A.
    # comm - MPI communicator linking the shared-memory processes.
    # index_buffer - a shared-memory integer buffer to use when finding pivot indices.
    # rank - the rank of this process in `comm`.
    # nproc - the number of processes in `comm`.

    # This function is essentially a copy of DGETRF2 from LAPACK, v3.12.1.
    # Recurse not over rows/columns but by splitting the matrix approximately in half each
    # step.

    @inbounds begin
        # Quick return if possible.
        if m == 0 || n == 0
            return nothing
        end

        comm = rplu.comm
        rank = rplu.rank
        nproc = rplu.nproc
        synchronize = rplu.synchronize

        if m * n < serial_threshold
            # For small (sub-)matrices, revert to a serial solve.
            if rank == 0
                #recursive_row_pivot_lu!(ipiv, A, m, n)
                getrf!(A, ipiv; check=false)
            end
            return nothing
        end

        if m == 1
            # One row case, just need to handle ipiv.
            if rank == 0
                ipiv[1] = 1
            end
        elseif n == 1
            # One column case.
            pivot_ind = find_pivot(rplu, @view(A[:,1]), m)
            if rank == 0
                ipiv[1] = pivot_ind

                # Apply the interchange
                A[1,1], A[pivot_ind,1] = A[pivot_ind,1], A[1,1]
            end

            synchronize()

            # Update the column
            rows_per_proc = (m - 1 + nproc - 1) ÷ nproc
            row_range = rank*rows_per_proc+2:min((rank+1)*rows_per_proc+1,m)
            @views A[row_range,1] .*= 1.0 / A[1,1]
        else
            n_diag = min(m, n)

            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            n1 = n_diag ÷ 2
            n2 = n - n1
            m2 = m - n1

            # Factor
            # [ A11 ]
            # [ --- ]
            # [ A21 ]
            recursive_row_pivot_lu!(rplu, @view(A[:,1:n1]), ipiv, m, n1)

            synchronize()

            # Apply interchanges to
            # [ A12 ]
            # [ --- ]
            # [ A22 ]
            cols_per_proc = (n2 + nproc - 1) ÷ nproc
            col_range = rank*cols_per_proc+n1+1:min((rank+1)*cols_per_proc+n1,n)
            if !isempty(col_range)
                apply_row_swaps!(@view(A[:,col_range]), ipiv, length(col_range), n1)

                # Solve A12
                A12 = @view A[1:n1,col_range]
                @views trsm!('L', 'L', 'N', 'U', 1.0, A[1:n1,1:n1], A12)

                # Update A22
                A12 = @view A[1:n1,col_range]
                A21 = @view A[n1+1:m,1:n1]
                A22 = @view A[n1+1:m,col_range]
                mul!(A22, A21, A12, -1.0, 1.0)
            end

            synchronize()

            # Factor A22
            bottom_ipiv = @view ipiv[n1+1:n_diag]
            recursive_row_pivot_lu!(rplu, @view(A[n1+1:m,n1+1:n]), bottom_ipiv, m2, n2)

            synchronize()

            # Apply interchanges to A21.
            cols_per_proc = (n1 + nproc - 1) ÷ nproc
            col_range = rank*cols_per_proc+1:min((rank+1)*cols_per_proc,n1)
            if !isempty(col_range)
                apply_row_swaps!(@view(A[n1+1:m,col_range]), bottom_ipiv,
                                 length(col_range), min(m2,n2))
            end

            synchronize()
            if rank == 0
                bottom_ipiv .+= n1
            end
        end
    end
    return nothing
end

end
