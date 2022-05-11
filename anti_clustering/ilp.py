from typing import List
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from scipy.spatial import distance_matrix
from ._base import AntiClustering

from anti_clustering.union_find import UnionFind


class ILPAntiClustering(AntiClustering):
    def run(self, df: pd.DataFrame, similarity_columns: List[str], num_groups: int, destination_column: str) -> pd.DataFrame:
        df = df.copy()

        data = df[similarity_columns].to_numpy()
        solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")

        min_group_size = np.floor(len(df)/num_groups)
        max_group_size = np.ceil(len(df)/num_groups)

        d = distance_matrix(data, data)
        x = np.asarray([[(solver.BoolVar(f'x_[{i}][{j}]')) for i in range(len(d))] for j in range(len(d))])

        for i in range(len(d)):
            for j in range(i+1, len(d)):
                for k in range(j+1, len(d)):
                    solver.Add(-x[i][j] + x[i][k] + x[j][k] <= 1)
                    solver.Add(x[i][j] - x[i][k] + x[j][k] <= 1)
                    solver.Add(x[i][j] + x[i][k] - x[j][k] <= 1)

        for i in range(len(d)):
            if i+1 < len(d):
                solver.Add(sum([x[i][j] for j in range(i+1, len(d))]) + sum([x[k][i] for k in range(0, i)]) <= max_group_size-1)
            if i > 0:
                solver.Add(sum([x[i][j] for j in range(i+1, len(d))]) + sum([x[k][i] for k in range(0, i)]) >= min_group_size-1)

        solver.Maximize(np.multiply(x, d).sum())

        status = solver.Solve()

        if status != 0:
            raise ValueError('Optimization failed!')

        result = np.asarray([[x[i][j].solution_value() for i in range(len(d))] for j in range(len(d))])

        components = UnionFind()
        components.initialize(range(len(d)))

        for i in range(len(d)):
            for j in range(0, i):
                if result[i][j] == 1:
                    components.union(i, j)

        df[destination_column] = [components.find(i) for i in range(len(d))]

        return df
