
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include <Kokkos_Core.hpp>
#include <KokkosSparse_spmv.hpp>


/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {
  Kokkos::Profiling::pushRegion("Optimized: ComputeSPMV");

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = true;
//  int result = ComputeSPMV_ref(A, x, y);
  Kokkos::View<const double*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > v_x(x.values,A.localNumberOfColumns);
  Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > v_y(y.values,A.localNumberOfRows);
  KokkosSparse::spmv("N",1.0,A.localMatrix,x.view,0.0,y.view);
  Kokkos::Profiling::popRegion();
  return 0;
}
