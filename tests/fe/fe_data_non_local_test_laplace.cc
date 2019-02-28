
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include "../tests.h"

using namespace dealii;

class Step3
{
public:
  Step3();

  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;
  MappingQ1<2>     mapping;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;


  Vector<double> solution;
  Vector<double> system_rhs;
};

Step3::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}

void Step3::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(5);

  deallog << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  deallog << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

// define the right hand-side function

template <int dim>
double function (const Point<dim> &p){
    return {- 8.0 * dealii::numbers::PI * dealii::numbers::PI
    * (sin(2.0*dealii::numbers::PI * p[0])
       * sin(2.0*dealii::numbers::PI * p[1]))};
}


void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(2);
    auto qpts = quadrature_formula.get_points();
    
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {

      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *
                 fe_values.shape_grad(j, q_index) *
                 fe_values.JxW(q_index));
          auto qpt = mapping.transform_unit_to_real_cell(cell, qpts[q_index]);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                            function(qpt) *
                            fe_values.JxW(q_index));
        }

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}


void Step3::solve()
{
  SolverControl solver_control(1000, 1e-12);

  SolverCG<> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  deallog <<  solution << std::endl;
}

void Step3::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
}

int main()
{
  initlog();
  
  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();

  return 0;
}
