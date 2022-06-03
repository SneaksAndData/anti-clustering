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
A simulated annealing approach to solving the anti-clustering problem.
"""

import math
from typing import List, Optional
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from anti_clustering._base import AntiClustering
from anti_clustering.union_find import UnionFind


class SimulatedAnnealingHeuristicAntiClustering(AntiClustering):
    def __init__(self, verbose: bool = False, alpha: float = 0.8, iterations: int = 200, starting_temperature: float = 10):
        super(SimulatedAnnealingHeuristicAntiClustering, self).__init__(verbose)
        self.alpha = alpha
        self.iterations = iterations
        self.starting_temperature = starting_temperature

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

        if self.verbose:
            print("Making costs")

        c = self._calculate_categorical_distance(df, categorical_columns) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
        d = self._calculate_numeric_distance(data) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)

        if self.verbose:
            print("Initializing clusters")

        initial_clusters = [i % num_groups for i in range(len(d)-num_groups)]
        rnd = random.Random(1)
        rnd.shuffle(initial_clusters)
        initial_clusters = [i for i in range(num_groups)] + initial_clusters
        uf_init = UnionFind()
        uf_init.initialize(initial_clusters)

        if self.verbose:
            print("Making vars")

        x = np.array([[uf_init.connected(i, j) for i in range(len(d))] for j in range(len(d))])

        if self.verbose:
            print("Solving")

        temperature = self.starting_temperature
        obj = self.calculate_objective(x, c, d)
        for iteration in range(self.iterations):
            if self.verbose and iteration % 5 == 0:
                print(f'{iteration} of {self.iterations}')

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

            temperature = temperature*self.alpha

        if self.verbose:
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
