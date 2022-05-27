from typing import List, Optional
import pandas as pd
import random
import numpy as np
from ._base import AntiClustering
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

        data = df[numeric_columns].to_numpy()

        print("Making costs")
        c = self._calculate_categorical_distance(df, categorical_columns) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)
        d = self._calculate_numeric_distance(data) if categorical_columns is not None and len(categorical_columns) > 0 else np.full(len(df), 0.0)

        print("Initializing clusters")
        initial_clusters = [i % num_groups for i in range(len(d))]
        rnd = random.Random(1)
        rnd.shuffle(initial_clusters)

        df[destination_column] = initial_clusters

        return df
