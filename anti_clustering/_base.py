# Copyright 2022 ECCO Sneaks & Data
#
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
import numpy.typing as npt
import pandas as pd
import scipy.spatial
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler
from anti_clustering._union_find import UnionFind


class AntiClustering(ABC):
    """Generic anti-clustering interface."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def run(
        self,
        df: pd.DataFrame,
        numerical_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        num_groups: int,
        destination_column: str,
    ) -> pd.DataFrame:
        """
        Run anti clustering algorithm on dataset.
        :param df: The dataset to run anti-clustering on.
        :param numerical_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :param num_groups: Number of anti-clusters to generate.
        :param destination_column: The column to write results to.
        :return: The original dataframe with a destination_column added.
        """
        numerical_columns = [] if numerical_columns is None else numerical_columns
        categorical_columns = [] if categorical_columns is None else categorical_columns

        prepared_df = self._prepare_data(
            df=df, numerical_columns=numerical_columns, categorical_columns=categorical_columns
        )

        distance_matrix = self._get_distance_matrix(
            df=prepared_df, numerical_columns=numerical_columns, categorical_columns=categorical_columns
        )

        cluster_assignment = self._solve(distance_matrix=distance_matrix, num_groups=num_groups)

        return self._post_process(
            df=df,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            destination_column=destination_column,
            cluster_assignment_matrix=cluster_assignment,
        )

    @abstractmethod
    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        """
        Abstract solve signature. To be implemented in subclasses.
        :param distance_matrix: The distance matrix of elements.
        :param num_groups: Number of anti-clusters to generate.
        :return:
        """

    def _prepare_data(
        self, df: pd.DataFrame, numerical_columns: List[str], categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Prepare data for solving.
        :param df: The input dataframe.
        :param numerical_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :return: the prepared dataframe.
        """
        if numerical_columns is None and categorical_columns is None:
            raise ValueError("Both numerical and categorical columns cannot be None.")

        # Normalize to interval [0, 1]
        if len(numerical_columns) > 0:
            scaler = MinMaxScaler()
            transformed_columns = scaler.fit_transform(df[numerical_columns])
            df = df.assign(**{col: transformed_columns[:, i] for i, col in enumerate(numerical_columns)})

        return df

    def _post_process(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        destination_column: str,
        cluster_assignment_matrix: npt.NDArray[bool],
    ) -> pd.DataFrame:
        # pylint: disable = R0913
        """
        Postprocess results and prepare for returning to caller.
        :param df: The input dataframe.
        :param numerical_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :param destination_column: The column to write results to.
        :param cluster_assignment_matrix: A matrix containing for each pair of elements if they belong to the same
        anti-cluster.
        :return: The inputted dataframe with the new destination column.
        """
        components = UnionFind(len(df))

        for j in range(len(df)):
            for i in range(0, j):
                if cluster_assignment_matrix[i][j] == 1:
                    components.union(i, j)

        df = df.assign(**{destination_column: [components.find(i) for i in range(len(df))]})

        # Normalize cluster labels. The algorithm assignment of cluster labels may be non-deterministic.
        # Ensure that all labels are enumerated starting from 0 without gaps.
        cluster_labels = df.sort_values(by=[*numerical_columns, *categorical_columns])[destination_column].unique()
        mapping = {k: i for i, k in enumerate(cluster_labels)}
        df = df.replace({destination_column: mapping})

        return df

    def _get_distance_matrix(
        self, df: pd.DataFrame, numerical_columns: List[str], categorical_columns: List[str]
    ) -> npt.NDArray[float]:
        """
        Calculate distance matrix between each pair of elements. Numeric columns default to Euclidean distance and
        categorical columns default to Hamming distance.
        :param df: The input dataframe.
        :param numerical_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :return: The distance matrix.
        """

        categorical_distance = 0
        if len(categorical_columns) > 0:
            categorical_distance = squareform(
                pdist(df[categorical_columns].apply(lambda x: pd.factorize(x)[0]), metric="hamming")
            )

        numeric_distance = 0
        if len(numerical_columns) > 0:
            numerical_data = df[numerical_columns].to_numpy()
            numeric_distance = scipy.spatial.distance_matrix(numerical_data, numerical_data)

        return numeric_distance + categorical_distance
