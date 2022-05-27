import math
from typing import List, Optional
import pandas as pd
import random
import numpy as np
from ._base import AntiClustering
from sklearn.preprocessing import MinMaxScaler

from anti_clustering.union_find import UnionFind

ITERATIONS = 100
ALPHA = 0.2

class SimulatedAnnealingHeuristicAntiClustering(AntiClustering):
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

        print("Making costs")
        c = self._calculate_categorical_distance(df, categorical_columns) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
        d = self._calculate_numeric_distance(data) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)

        print("Initializing clusters")
        initial_clusters = [i % num_groups for i in range(len(d)-num_groups)]
        rnd = random.Random(1)
        rnd.shuffle(initial_clusters)
        initial_clusters = [i for i in range(num_groups)] + initial_clusters
        uf_init = UnionFind()
        uf_init.initialize(initial_clusters)

        print("Making vars")
        x = np.array([[uf_init.connected(i, j) for i in range(len(d))] for j in range(len(d))])

        print("Solving")
        temperature = 10
        obj = self.calculate_objective(x, c, d)
        for iteration in range(ITERATIONS):
            if iteration % 5 == 0:
                print(f'{iteration} of {ITERATIONS}')

            # generate neighbor
            i = rnd.randint(0, len(d)-1)
            possible_exchanges = self.get_exchanges(x, i)
            if len(possible_exchanges) == 0:
                continue
            j = possible_exchanges[rnd.randint(0, len(possible_exchanges)-1)]

            new_x = self.swap(x, i, j)
            new_obj = self.calculate_objective(new_x, c, d)

            if self.accept(new_obj - obj, temperature, rnd):
                obj = new_obj
                x = new_x

            temperature = temperature*ALPHA


        print("Unioning clusters")
        components = UnionFind()
        components.initialize(range(len(d)))

        for i in range(len(d)):
            for j in range(0, i):
                if x[i][j]:
                    components.union(i, j)

        df[destination_column] = [components.find(i) for i in range(len(d))]

        return df


    def swap(self, matrix, i, j):
        matrix = matrix.copy()
        tmp1 = matrix[i,].copy()
        tmp2 = matrix[:, i].copy()
        matrix[i,] = matrix[j,]
        matrix[:, i] = matrix[:, j]
        matrix[j,] = tmp1
        matrix[:, j] = tmp2
        matrix[i, j] = False
        matrix[j, i] = False
        matrix[i, i] = True
        matrix[j, j] = True
        return matrix

    def calculate_objective(self, x, c, d):
        return np.multiply(x, c + d).sum()

    def accept(self, d, t, r):
        return d >= 0 or math.exp(d/t) >= r.uniform(0, 1)

    def get_exchanges(self, matrix, i):
        return np.nonzero(np.invert(matrix[i]))[0]

    # def get_neighbors(self, i, j, max_index):
    #     return [
    #         (i+di, j+dj) for di in range(-1,2) for dj in range(-1,2)
    #         if 0 <= i + di <= max_index and 0 <= j + dj <= max_index
    #     ]
