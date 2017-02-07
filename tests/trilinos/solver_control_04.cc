// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// test that we can reuse a Trilinos solver in a LinearOperator


#include "../tests.h"
#include <deal.II/base/utilities.h>
#include "../testmatrix.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

int main(int argc, char **argv)
{
  std::ofstream logfile("output");
  logfile.precision(4);
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, testing_max_num_threads());


  {
    const unsigned int size = 32;
    unsigned int dim = (size-1)*(size-1);

    deallog << "Size " << size << " Unknowns " << dim << std::endl;

    // Make matrix
    FDMatrix testproblem(size, size);
    DynamicSparsityPattern csp (dim, dim);
    testproblem.five_point_structure(csp);
    TrilinosWrappers::SparseMatrix  A;
    A.reinit(csp);
    testproblem.five_point(A);

    TrilinosWrappers::Vector  f(dim);
    TrilinosWrappers::Vector  u1(dim);
    TrilinosWrappers::Vector  u2(dim);
    f = 1.;
    A.compress (VectorOperation::insert);
    f.compress (VectorOperation::insert);
    u1.compress (VectorOperation::insert);
    u2.compress (VectorOperation::insert);

    TrilinosWrappers::PreconditionJacobi preconditioner;
    preconditioner.initialize(A);

    SolverControl solver_control(2000, 1.e-3);
    TrilinosWrappers::SolverCG solver(solver_control);

    const auto lo_A = linear_operator<TrilinosWrappers::Vector>(A);
    const auto lo_A_inv = inverse_operator(lo_A,
                                           solver,
                                           preconditioner);

    deallog.push("First use");
    {
      u1 = lo_A_inv * f;
      deallog << "OK" << std::endl;
    }
    deallog.pop();
    deallog.push("Second use");
    {
      u2 = lo_A_inv * f;
      deallog << "OK" << std::endl;
    }
    deallog.pop();
  }

  deallog << "OK" << std::endl;
}