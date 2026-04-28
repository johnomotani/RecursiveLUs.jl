# MPISharedMemLUs

[![Build Status](https://github.com/johnomotani/MPISharedMemLUs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/johnomotani/MPISharedMemLUs.jl/actions/workflows/CI.yml?query=branch%3Amain)

The package provides a variant of the
[LAPACK](https://github.com/Reference-LAPACK/lapack/) LU factorisation algorithm that
uses column pivoting instead of row pivoting.

A direct copy of the standard (row-pivoting) algorithm is also included to enable a fair
performance comparison.

The column-pivoting variant is less efficient (maybe ~2x), so should be used only for
special applications where column pivoting is required.

