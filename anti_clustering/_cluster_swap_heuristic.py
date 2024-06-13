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
"""Abstract class containing utilities for cluster swap-based heuristics."""

from abc import ABC
import random
import numpy as np
import numpy.typing as npt
from anti_clustering._base import AntiClustering
from anti_clustering._union_find import UnionFind


class ClusterSwapHeuristic(AntiClustering, ABC):
    """Abstract class containing utilities for cluster swap-based heuristics."""

    def __init__(self, verbose: bool = False, random_seed: int = None):
        super().__init__(verbose=verbose)
        self.rnd = random.Random(random_seed)

    def _get_exchanges(self, cluster_assignment: npt.NDArray[bool], i: int) -> npt.NDArray[int]:
        """
        Given a cluster assignment matrix and element index, will return possible indexes to swap anti-clusters with.
        :param cluster_assignment: Cluster assignment matrix.
        :param i: Element index.
        :return: Possible exchanges.
        """
        return np.nonzero(np.invert(cluster_assignment[i]))[0]

    def _swap(self, cluster_assignment: npt.NDArray[bool], i: int, j: int) -> npt.NDArray[bool]:
        """
        Swap anti-clusters of elements i and j.
        :param cluster_assignment: Current cluster assignment.
        :param i: Element.
        :param j: Other element.
        :return: Cluster assignment with i and j swapped.
        """
        cluster_assignment = cluster_assignment.copy()
        tmp1 = cluster_assignment[i,].copy()
        tmp2 = cluster_assignment[:, i].copy()
        cluster_assignment[i,] = cluster_assignment[j,]
        cluster_assignment[:, i] = cluster_assignment[:, j]
        cluster_assignment[j,] = tmp1
        cluster_assignment[:, j] = tmp2
        cluster_assignment[i, j] = False
        cluster_assignment[j, i] = False
        return cluster_assignment

    def _get_random_clusters(self, num_groups: int, num_elements: int) -> npt.NDArray[bool]:
        """
        Get a random initialization of anti-clusters.
        :param num_groups: Number of anti-clusters to generate.
        :param num_elements: Number of elements in algorithm run.
        :return: The randomly initialized anti-clusters as an assignment matrix
        """
        if self.verbose:
            print("Initializing clusters")

        # Using UnionFind to generate random anti-clusters. The first num_groups elements are guaranteed
        # to be roots of each their own component. All other elements are assigned a random root.
        initial_clusters = [i % num_groups for i in range(num_elements - num_groups)]
        self.rnd.shuffle(initial_clusters)
        initial_clusters = list(range(num_groups)) + initial_clusters
        uf_init = UnionFind(len(initial_clusters))  # pylint: disable = R1721
        for i, cluster in enumerate(initial_clusters):
            uf_init.union(i, cluster)

        cluster_assignment = np.array(
            [[uf_init.connected(i, j) for i in range(num_elements)] for j in range(num_elements)]
        )

        return cluster_assignment

    def _calculate_objective(self, cluster_assignment: npt.NDArray[bool], distance_matrix: npt.NDArray[float]) -> float:
        """
        Calculate objective value
        :param cluster_assignment: Cluster assignment
        :param distance_matrix: Distance matrix
        :return: Objective value
        """
        return np.multiply(cluster_assignment, distance_matrix).sum()
