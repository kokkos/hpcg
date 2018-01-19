
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include "KokkosBlas1_update.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = true;
  Kokkos::View<const double*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > v_x(x.values,n);
  Kokkos::View<const double*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > v_y(y.values,n);
  Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > v_w(w.values,n);
  KokkosBlas::update(alpha,v_x,beta,v_y,0.0,v_w);
  return 0;
}
