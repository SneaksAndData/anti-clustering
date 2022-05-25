from typing import List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance_matrix


class AntiClustering(ABC):
    @abstractmethod
    def run(self, df: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str], num_groups: int, destination_column: str)  -> pd.DataFrame:
        pass

    def _calculate_categorical_distance(self, df: pd.DataFrame, categorical_columns: List[str]):
        return squareform(pdist(df[categorical_columns].apply(lambda x: pd.factorize(x)[0]), metric='hamming'))

    def _calculate_numeric_distance(self, data: np.ndarray):
        return distance_matrix(data, data)
