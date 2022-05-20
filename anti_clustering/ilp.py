from typing import List, Optional
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform, pdist
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

        c = self._calculate_categorical_distance(df, categorical_columns) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
        d = distance_matrix(data, data) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
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

        solver.Maximize(np.multiply(x, d).sum() + np.multiply(x, c).sum())

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

    def _calculate_categorical_distance(self, df: pd.DataFrame, categorical_columns: List[str]):
        return squareform(pdist(df[categorical_columns].apply(lambda x: pd.factorize(x)[0]), metric='hamming'))
