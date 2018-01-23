
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#include <KokkosSparse_gauss_seidel.hpp>

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  LocalSparseMatrix local_matrix = A.localMatrix;
  const int apply_count = 1;

  if(!A.coloring_done) {
    KokkosSparse::Experimental::gauss_seidel_symbolic
      (&A.kh, A.localNumberOfRows, A.localNumberOfColumns, A.localMatrix.graph.row_map, A.localMatrix.graph.entries, true);
    A.coloring_done = true;
  }

  KokkosSparse::Experimental::gauss_seidel_numeric
    (&A.kh, A.localNumberOfRows, A.localNumberOfColumns, A.localMatrix.graph.row_map, A.localMatrix.graph.entries, A.localMatrix.values, true);

  KokkosSparse::Experimental::symmetric_gauss_seidel_apply
    (&A.kh, A.localNumberOfRows, A.localNumberOfColumns, A.localMatrix.graph.row_map, A.localMatrix.graph.entries, A.localMatrix.values, 
     x.view,r.view);
     //Kokkos::View<double*>(x.view,std::pair<size_t,size_t>(0,num_cols_1)), Kokkos::View<const double*>(r.view,std::pair<size_t,size_t>(0,num_rows_1)));

  return 0;
}
