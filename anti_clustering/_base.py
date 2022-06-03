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
"""Generic anti-clustering interface."""

from typing import List, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance_matrix


class AntiClustering(ABC):
    def __init__(self, verbose=False):
        self.verbose = verbose

    @abstractmethod
    def run(self, df: pd.DataFrame, numeric_columns: Optional[List[str]], categorical_columns: Optional[List[str]], num_groups: int, destination_column: str)  -> pd.DataFrame:
        pass

    def _calculate_categorical_distance(self, df: pd.DataFrame, categorical_columns: List[str]):
        return squareform(pdist(df[categorical_columns].apply(lambda x: pd.factorize(x)[0]), metric='hamming'))

    def _calculate_numeric_distance(self, data: np.ndarray):
        return distance_matrix(data, data)
