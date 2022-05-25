from typing import List, Optional, Union, Iterable
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from sklearn.preprocessing import MinMaxScaler
from ._base import AntiClustering


from anti_clustering.union_find import UnionFind


class ILPAntiClustering(AntiClustering):
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
        solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")

        min_group_size = np.floor(len(df)/num_groups)
        max_group_size = np.ceil(len(df)/num_groups)

        print("Making costs")
        c = self._calculate_categorical_distance(df, categorical_columns) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
        d = self._calculate_numeric_distance(data) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)

        print("Making vars")
        x = np.asarray([[(solver.BoolVar(f'x_[{i}][{j}]')) for i in range(len(d))] for j in range(len(d))])

        print("Making cluster assignment constrs")
        for i in range(len(d)):
            for j in range(i+1, len(d)):
                for k in range(j+1, len(d)):
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

        print("Making obj")
        solver.Maximize(np.multiply(x, d).sum() + np.multiply(x, c).sum())

        print("Solving")
        status = solver.Solve()

        if status != 0:
            raise ValueError('Optimization failed!')

        result = np.asarray([[x[i][j].solution_value() for i in range(len(d))] for j in range(len(d))])

        print("Unioning clusters")
        components = UnionFind()
        components.initialize(range(len(d)))

        for i in range(len(d)):
            for j in range(0, i):
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
