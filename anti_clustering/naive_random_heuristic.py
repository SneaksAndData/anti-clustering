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
The naive randomized way of solving the anti-clustering problem.
"""

from typing import List, Optional
import pandas as pd
import random
from anti_clustering._base import AntiClustering
from sklearn.preprocessing import MinMaxScaler


class NaiveRandomHeuristicAntiClustering(AntiClustering):
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

        initial_clusters = [i % num_groups for i in range(len(df))]
        rnd = random.Random(1)
        rnd.shuffle(initial_clusters)

        df[destination_column] = initial_clusters

        return df
