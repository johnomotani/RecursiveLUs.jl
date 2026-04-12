module ColumnPivotLUs

export row_pivot_lu!, column_pivot_lu!

#using LoopVectorization
using LinearAlgebra
using LinearAlgebra.BLAS: trsm!

# This is LAPACK's default block size for DGETRF()
const block_size = 64

function find_pivot(a, n)
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

function apply_column_swaps!(A, jpiv, m, npivot)
    for j ∈ 1:npivot
        pivot_ind = jpiv[j]
        for i ∈ 1:m
            A[i,j], A[i,pivot_ind] = A[i,pivot_ind], A[i,j]
        end
    end
    return nothing
end

function column_pivot_lu!(A::AbstractMatrix, jpiv::AbstractVector{<:Integer})
    blocked_column_pivot_lu!(A, jpiv, size(A, 1), size(A, 2))
    return A
end

function blocked_column_pivot_lu!(A::AbstractMatrix, jpiv::AbstractVector{<:Integer},
                                  m::Integer, n::Integer)
    @inbounds begin
        n_diag = min(m, n)

        if n_diag ≤ block_size
            return recursive_column_pivot_lu!(A, jpiv, m, n)
        end

        for i ∈ 1:block_size:n_diag
            ib = min(block_size, n_diag - i + 1)
            ie = i + ib - 1
            this_jpiv = @view jpiv[i:n_diag]

            # Factor diagonal and right-of-diagonal blocks.
            @views recursive_column_pivot_lu!(A[i:ie,i:n], this_jpiv, ib, n - i + 1)

            # Apply interchanges to rows 1:i-1
            if i > 1
                apply_column_swaps!(@view(A[1:i-1,i:n]), this_jpiv, i - 1, ib)
            end

            if i + ib ≤ m
                m2 = m - ie
                n2 = n - ie

                # Apply interchanges to rows i+ib:m
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

function recursive_column_pivot_lu!(A::AbstractMatrix, jpiv::AbstractVector{<:Integer},
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
            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            m1 = min(m, n) ÷ 2
            m2 = m - m1
            n2 = n - m1

            # Factor
            # [ A11 | A12 ]
            recursive_column_pivot_lu!(@view(A[1:m1,:]), jpiv, m1, n)

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
            right_jpiv = @view jpiv[m1+1:min(m,n)]
            recursive_column_pivot_lu!(A22, right_jpiv, m2, n2)

            # Apply interchanges to A12
            apply_column_swaps!(A12, right_jpiv, m1, min(m2,n2))

            right_jpiv .+= m1
        end
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
                for k ∈ j:n
                    A[i,k], A[pivot,k] = A[pivot,k], A[i,k]
                end
            end
        end
    end
    return nothing
end

function row_pivot_lu!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer})
    blocked_row_pivot_lu!(A, ipiv, size(A, 1), size(A, 2))
    return A
end

function blocked_row_pivot_lu!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer},
                               m::Integer, n::Integer)
    @inbounds begin
        n_diag = min(m, n)

        if n_diag ≤ block_size
            return recursive_row_pivot_lu!(A, ipiv, m, n)
        end

        for j ∈ 1:block_size:n_diag
            jb = min(block_size, n_diag - j + 1)
            je = j + jb - 1
            this_ipiv = @view ipiv[j:n_diag]

            # Factor diagonal and subdiagonal blocks.
            @views recursive_row_pivot_lu!(A[j:m,j:je], this_ipiv, m - j + 1, jb)

            # Apply interchanges to columns 1:j-1
            if j > 1
                apply_row_swaps!(@view(A[j:m,1:j-1]), this_ipiv, j - 1, jb)
            end

            if j + jb ≤ n
                m2 = m - je
                n2 = n - je

                # Apply interchanges to columns j+jb:n
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

function recursive_row_pivot_lu!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer},
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
            # Block-factorise
            # [ A11 | A12 ]
            # [ --------- ]
            # [ A21 | A22 ]
            n1 = min(m, n) ÷ 2
            n2 = n - n1
            m2 = m - n1

            # Factor
            # [ A11 ]
            # [ --- ]
            # [ A21 ]
            recursive_row_pivot_lu!(@view(A[:,1:n1]), ipiv, m, n1)

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
            bottom_ipiv = @view ipiv[n1+1:min(m,n)]
            recursive_row_pivot_lu!(A22, bottom_ipiv, m2, n2)

            # Apply interchanges to A21
            apply_row_swaps!(A21, bottom_ipiv, n1, min(m2,n2))

            bottom_ipiv .+= n1
        end
    end
    return nothing
end

end
