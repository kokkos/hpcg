
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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction(const SparseMatrix & A, const Vector & rf) {

  Kokkos::View<double*> Axfv = A.mgData->Axf->view;
  Kokkos::View<double*> rfv = rf.view;
  Kokkos::View<double*> rcv = A.mgData->rc->view;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  Kokkos::parallel_for("ComputeRestriction",nc, KOKKOS_LAMBDA (const local_int_t& i) {
    rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  });
  return 0;
}
