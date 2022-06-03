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
The exchange heuristic to solving the anti-clustering problem.

Based on:
Papenberg, M., & Klau, G. W. (2021). Using anticlustering to partition data sets into equivalent parts.
Psychological Methods, 26(2), 161–174. https://doi.org/10.1037/met0000301
"""

from typing import List, Optional
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from anti_clustering._base import AntiClustering
from anti_clustering.union_find import UnionFind


class ExchangeHeuristicAntiClustering(AntiClustering):
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

        current_objective = self.calculate_objective(x, c, d)
        for i in range(len(d)):
            if self.verbose and i % 5 == 0:
                print(f'{i} of {len(d)}')
            exchange_indices = self.get_exchanges(x, i)
            if len(exchange_indices) == 0:
                continue
            exchanges = [
                (self.calculate_objective(self.swap(x, i, j), c, d), j) for j in exchange_indices
            ]
            best_exchange = max(exchanges)
            if best_exchange[0] > current_objective:
                x = self.swap(x, i, best_exchange[1])
                current_objective = best_exchange[0]

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

    def get_exchanges(self, matrix, i):
        return np.nonzero(np.invert(matrix[i]))[0]

    def swap(self, matrix, i, j):
        matrix = matrix.copy()
        tmp1 = matrix[i, ].copy()
        tmp2 = matrix[:, i].copy()
        matrix[i, ] = matrix[j, ]
        matrix[:, i] = matrix[:, j]
        matrix[j, ] = tmp1
        matrix[:, j] = tmp2
        matrix[i, j] = False
        matrix[j, i] = False
        return matrix

    def calculate_objective(self, x, c, d):
        return np.multiply(x, c + d).sum()
