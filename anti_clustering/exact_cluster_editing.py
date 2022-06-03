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

from typing import List, Optional, Union, Iterable
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from sklearn.preprocessing import MinMaxScaler
from anti_clustering._base import AntiClustering
from anti_clustering.union_find import UnionFind


class ExactClusterEditingAntiClustering(AntiClustering):
    def run(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        num_groups: int,
        destination_column: str
    ) -> pd.DataFrame:
        if numeric_columns is None and categorical_columns is None:
            raise ValueError('Both numeric and categorical columns cannot be None.')

        df = df.copy()

        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        data = df[numeric_columns].to_numpy()
        solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("GLOP")

        if self.verbose:
            solver.EnableOutput()

        min_group_size = np.floor(len(df)/num_groups)
        max_group_size = np.ceil(len(df)/num_groups)

        if self.verbose:
            print("Making costs")

        c = self._calculate_categorical_distance(df, categorical_columns) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
        d = self._calculate_numeric_distance(data) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)

        if self.verbose:
            print("Making vars")

        x = np.asarray([[(solver.BoolVar(f'x_[{i}][{j}]')) if j > i else False for j in range(len(d))] for i in range(len(d))])

        if self.verbose:
            print("Making cluster assignment constrs")

        for k in range(len(d)):
            for j in range(0, k):
                for i in range(0, j):
                    self.add_constraint(
                        solver=solver,
                        vars_=[x[i][j], x[i][k], x[j][k]],
                        coeffs=[-1.0, 1.0, 1.0],
                        ub=1.0,
                        lb=-solver.infinity()
                    )

                    self.add_constraint(
                        solver=solver,
                        vars_=[x[i][j], x[i][k], x[j][k]],
                        coeffs=[1.0, -1.0, 1.0],
                        ub=1.0,
                        lb=-solver.infinity()
                    )

                    self.add_constraint(
                        solver=solver,
                        vars_=[x[i][j], x[i][k], x[j][k]],
                        coeffs=[1.0, 1.0, -1.0],
                        ub=1.0,
                        lb=-solver.infinity()
                    )

        if self.verbose:
            print("Making cluster size constrs")

        for i in range(len(d)):
            if i+1 < len(d):
                self.add_constraint(
                    solver=solver,
                    vars_=[x[i][j] for j in range(i+1, len(d))] + [x[k][i] for k in range(0, i)],
                    coeffs=[1.0 for j in range(i+1, len(d))] + [1.0 for k in range(0, i)],
                    ub=max_group_size-1.0,
                    lb=-solver.infinity()
                )

            if i > 0:
                self.add_constraint(
                    solver=solver,
                    vars_=[x[i][j] for j in range(i+1, len(d))] + [x[k][i] for k in range(0, i)],
                    coeffs=[1.0 for j in range(i+1, len(d))] + [1.0 for k in range(0, i)],
                    lb=min_group_size - 1.0,
                    ub=solver.infinity()
                )

        if self.verbose:
            print("Making obj")

        solver.Maximize(np.multiply(x, c + d).sum())

        if self.verbose:
            print("Solving")

        status = solver.Solve()

        if status != 0:
            raise ValueError('Optimization failed!')

        result = np.asarray([[x[i][j].solution_value() if j > i else None for j in range(len(d))] for i in range(len(d))])

        if self.verbose:
            print("Unioning clusters")

        components = UnionFind()
        components.initialize(range(len(d)))

        for j in range(len(d)):
            for i in range(0, j):
                if result[i][j] == 1:
                    components.union(i, j)

        df[destination_column] = [components.find(i) for i in range(len(d))]

        return df

    def add_constraint(self, solver, lb: float, ub: float, coeffs: Union[List[float], float], vars_: Union[List[pywraplp.Variable], pywraplp.Variable]) -> pywraplp.Constraint:
        #TODO: protect against bad inputs
        constr: pywraplp.Constraint = solver.Constraint(lb, ub)

        if isinstance(vars_, Iterable):
            for (coeff, var) in zip(coeffs, vars_):
                constr.SetCoefficient(var, coeff)
        else:
            constr.SetCoefficient(vars_, coeffs)

        return constr
