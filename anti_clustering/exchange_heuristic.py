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

import numpy as np
import numpy.typing as npt
from anti_clustering._cluster_swap_heuristic import ClusterSwapHeuristic


class ExchangeHeuristicAntiClustering(ClusterSwapHeuristic):
    """
    The exchange heuristic to solving the anti-clustering problem.
    """
    def __init__(self, verbose: bool = False, random_seed: int = None):
        super().__init__(verbose=verbose, random_seed=random_seed)

    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        # Starts with random cluster assignment
        cluster_assignment = self._get_random_clusters(num_groups=num_groups, num_elements=len(distance_matrix))

        if self.verbose:
            print("Solving")

        # Initial objective value
        current_objective = self._calculate_objective(cluster_assignment, distance_matrix)
        for i in range(len(distance_matrix)):
            if self.verbose and i % 5 == 0:
                print(f'Iteration {i + 1} of {len(distance_matrix)}')

            # Get list of possible swaps
            exchange_indices = self._get_exchanges(cluster_assignment, i)

            if len(exchange_indices) == 0:
                continue

            # Calculate objective value for all possible swaps.
            # List contains tuples of obj. val. and swapped element index.
            exchanges = [
                (self._calculate_objective(self._swap(cluster_assignment, i, j), distance_matrix), j) for j in exchange_indices
            ]

            # Find best swap
            best_exchange = max(exchanges)

            # If best swap is better than current objective value then complete swap
            if best_exchange[0] > current_objective:
                cluster_assignment = self._swap(cluster_assignment, i, best_exchange[1])
                current_objective = best_exchange[0]

        return cluster_assignment

    def _calculate_objective(self, cluster_assignment: npt.NDArray[bool], distance_matrix: npt.NDArray[float]) -> float:
        """
        Calculate objective value
        :param cluster_assignment: Cluster assignment matrix
        :param distance_matrix: Cost matrix
        :return: Objective value
        """
        return np.multiply(cluster_assignment, distance_matrix).sum()
