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
from anti_clustering.union_find import UnionFind


class AntiClustering(ABC):
    """Generic anti-clustering interface."""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def run(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        num_groups: int,
        destination_column: str
    ) -> pd.DataFrame:
        """
        Run anti clustering algorithm on dataset.
        :param df: The dataset to run anti-clustering on.
        :param numeric_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :param num_groups: Number of anti-clusters to generate.
        :param destination_column: The column to write results to.
        :return: The original dataframe with a destination_column added.
        """
        prepared_df = self._prepare_data(
            df=df,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )

        distance_matrix = self._get_distance_matrix(
            df=prepared_df,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )

        cluster_assignment = self._solve(distance_matrix=distance_matrix, num_groups=num_groups)

        return self._post_process(
            df=df,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            destination_column=destination_column,
            cluster_assignment_matrix=cluster_assignment
        )

    @abstractmethod
    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        """
        Abstract solve signature. To be implemented in subclasses.
        :param distance_matrix: The distance matrix of elements.
        :param num_groups: Number of anti-clusters to generate.
        :return:
        """
        pass

    def _prepare_data(self, df: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]) -> pd.DataFrame:
        """
        Prepare data for solving.
        :param df: The input dataframe.
        :param numeric_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :return: the prepared dataframe.
        """
        if numeric_columns is None and categorical_columns is None:
            raise ValueError('Both numeric and categorical columns cannot be None.')

        df = df.copy()

        # Normalize to interval [0, 1]
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        return df

    def _post_process(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        destination_column: str,
        cluster_assignment_matrix: npt.NDArray[bool]
    ) -> pd.DataFrame:
        """
        Postprocess results and prepare for returning to caller.
        :param df: The input dataframe.
        :param numeric_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :param destination_column: The column to write results to.
        :param cluster_assignment_matrix: A matrix containing for each pair of elements if they belong to the same anti-cluster.
        :return: The inputted dataframe with the new destination column.
        """
        components = UnionFind({i: i for i in range(len(df))})

        for j in range(len(df)):
            for i in range(0, j):
                if cluster_assignment_matrix[i][j] == 1:
                    components.union(i, j)

        df[destination_column] = [components.find(i) for i in range(len(df))]

        # Normalize cluster labels. The algorithm assignment of cluster labels may be non-deterministic.
        # Ensure that all labels are enumerated starting from 0 without gaps.
        cluster_labels = df.sort_values(by=[*numeric_columns, *categorical_columns])[destination_column].unique()
        mapping = {
            k: i for i, k in enumerate(cluster_labels)
        }
        df = df.replace({'Cluster': mapping})

        return df

    def _get_distance_matrix(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str]
    ) -> npt.NDArray[float]:
        """
        Calculate distance matrix between each pair of elements. Numeric columns default to Euclidean distance and
        categorical columns default to Hamming distance.
        :param df: The input dataframe.
        :param numeric_columns: Columns in dataset to use for anti-clustering containing numbers.
        :param categorical_columns: Columns in dataset to use for anti-clustering containing strings or dates.
        :return: The distance matrix.
        """

        d = squareform(pdist(df[categorical_columns].apply(lambda x: pd.factorize(x)[0]), metric='hamming'))

        numeric_data = df[numeric_columns].to_numpy()
        c = scipy.spatial.distance_matrix(numeric_data, numeric_data)

        return c + d
