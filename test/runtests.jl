using ColumnPivotLUs
using LinearAlgebra
using StableRNGs
using Test

function test_column_pivoting(m, n, tol)
    rng = StableRNG(234)
    @testset "column pivoting" begin
        A = rand(rng, m, n)
        s = min(m, n)
        jpiv = zeros(Int64, s)
        ALU = copy(A)
        column_pivot_lu!(ALU, jpiv)
        L = fill(NaN, m, s)
        L[1:s,1:s] .= UnitLowerTriangular(ALU[1:s,1:s])
        L[s+1:end,1:s] .= ALU[s+1:end,1:s]
        U = fill(NaN, s, n)
        U[1:s,1:s] .= UpperTriangular(ALU[1:s,1:s])
        U[1:s,s+1:end] .= ALU[1:s,s+1:end]
        p = LinearAlgebra.ipiv2perm(jpiv, n)
        @test isapprox(L * U, A[:,p], atol=tol, norm=x->NaN)
    end
    return nothing
end

function test_row_pivoting(m, n, tol)
    rng = StableRNG(123)
    @testset "row pivoting" begin
        A = rand(rng, m, n)
        s = min(m, n)
        ipiv = zeros(Int64, s);
        ALU = copy(A)
        row_pivot_lu!(ALU, ipiv)
        L = fill(NaN, m, s)
        L[1:s,1:s] .= UnitLowerTriangular(ALU[1:s,1:s])
        L[s+1:end,1:s] .= ALU[s+1:end,1:s]
        U = fill(NaN, s, n)
        U[1:s,1:s] .= UpperTriangular(ALU[1:s,1:s])
        U[1:s,s+1:end] .= ALU[1:s,s+1:end]
        p = LinearAlgebra.ipiv2perm(ipiv, m)
        @test isapprox(L * U, A[p,:], atol=tol, norm=x->NaN)
    end
    return nothing
end

@testset "ColumnPivotLUs.jl" begin
    tol = 4.0e-13
    @testset "m=$m n=$n" for m ∈ [16, 32, 53, 64, 128, 143, 4096], n ∈ [16, 32, 53, 64, 128, 143, 4096]
        if m > 2048 && n > 2048
            continue
        end
        test_column_pivoting(m, n, tol)
        test_row_pivoting(m, n, tol)
    end
end
