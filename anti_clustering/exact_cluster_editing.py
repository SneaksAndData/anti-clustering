# Copyright 2022 ECCO Sneaks & Data
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MIP formulation for solving the anti-clustering problem.

Based on:
Papenberg, M., & Klau, G. W. (2021). Using anticlustering to partition data sets into equivalent parts.
Psychological Methods, 26(2), 161–174. https://doi.org/10.1037/met0000301
"""

from typing import List
import numpy as np
import numpy.typing as npt
from ortools.linear_solver import pywraplp
from anti_clustering._base import AntiClustering


class ExactClusterEditingAntiClustering(AntiClustering):
    """
    MIP formulation for solving the anti-clustering problem.
    """
    def __init__(self, verbose: bool = False, solver_id: str = "SCIP"):
        super().__init__(verbose=verbose)
        self.solver_id = solver_id

    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        solver: pywraplp.Solver = pywraplp.Solver.CreateSolver(self.solver_id)

        if self.verbose:
            solver.EnableOutput()

        min_group_size = np.floor(len(distance_matrix) / num_groups)
        max_group_size = np.ceil(len(distance_matrix) / num_groups)

        # Cluster assignment are modelled as boolean assignments.
        x = np.asarray([
            [
                (solver.BoolVar(f'x_[{i}][{j}]'))
                if j > i else False
                for j in range(len(distance_matrix))
            ]
            for i in range(len(distance_matrix))])

        if self.verbose:
            print("Making cluster assignment constraints")

        # Cluster assignment constraints
        for k in range(len(distance_matrix)):
            for j in range(0, k):
                for i in range(0, j):
                    self._add_constraint(
                        solver=solver,
                        vars_=[x[i][j], x[i][k], x[j][k]],
                        coeffs=[-1.0, 1.0, 1.0],
                        ub=1.0,
                        lb=-solver.infinity()
                    )

                    self._add_constraint(
                        solver=solver,
                        vars_=[x[i][j], x[i][k], x[j][k]],
                        coeffs=[1.0, -1.0, 1.0],
                        ub=1.0,
                        lb=-solver.infinity()
                    )

                    self._add_constraint(
                        solver=solver,
                        vars_=[x[i][j], x[i][k], x[j][k]],
                        coeffs=[1.0, 1.0, -1.0],
                        ub=1.0,
                        lb=-solver.infinity()
                    )

        if self.verbose:
            print("Making cluster size constraints")

        # Cluster size constraints. Differently to original paper, we allow anti-clusters to be of different size if
        # number of groups does not divide number og elements.
        for i in range(len(distance_matrix)):
            self._add_constraint(
                solver=solver,
                vars_=[x[i][j] for j in range(i + 1, len(distance_matrix))] + [x[k][i] for k in range(0, i)],
                coeffs=[1.0 for j in range(i + 1, len(distance_matrix))] + [1.0 for k in range(0, i)],
                ub=max_group_size - 1.0 if i + 1 < len(distance_matrix) else solver.infinity(),
                lb=min_group_size - 1.0 if i > 0 else -solver.infinity()
            )

        if self.verbose:
            print("Making objective")

        # Maximise internal anti-cluster distance
        solver.Maximize(np.multiply(x, distance_matrix).sum())

        if self.verbose:
            print("Solving")

        status = solver.Solve()

        if status != 0:
            raise ValueError('Optimization failed!')

        cluster_assignment = np.asarray([
            [
                bool(x[i][j].solution_value())
                if j > i else None
                for j in range(len(distance_matrix))
            ]
            for i in range(len(distance_matrix))
        ])

        return cluster_assignment

    def _add_constraint(
        self,
        solver: pywraplp.Solver,
        lb: float,
        ub: float,
        coeffs: List[float],
        vars_: List[pywraplp.Variable]
    ) -> pywraplp.Constraint:
        # pylint: disable = R0201, R0913
        """
        Utility for adding constraints in the Google OR-Tools framework. Adds single constraint on the form:
        lb <= c_1x_1 + c_2x_2 + ... <= ub
        :param solver: The OR-Tools solver.
        :param lb: Lower bound.
        :param ub: Upper bound.
        :param coeffs: A list of coefficients. Each index must correspond to the same index in vars_.
        :param vars_: A list of decision variables. Each index must correspond to the same index in coeffs.
        :return: The OR-Tools constraint.
        """
        constr: pywraplp.Constraint = solver.Constraint(lb, ub)

        for (coeff, var) in zip(coeffs, vars_):
            constr.SetCoefficient(var, coeff)

        return constr
