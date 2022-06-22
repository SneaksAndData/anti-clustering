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
"""
The naive randomized way of solving the anti-clustering problem.
"""

import numpy.typing as npt
from anti_clustering._cluster_swap_heuristic import ClusterSwapHeuristic


class NaiveRandomHeuristicAntiClustering(ClusterSwapHeuristic):
    """
    The naive randomized way of solving the anti-clustering problem.
    """
    def __init__(self, verbose: bool = False, random_seed: int = None):
        super().__init__(verbose=verbose, random_seed=random_seed)

    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        return self._get_random_clusters(num_groups=num_groups, num_elements=len(distance_matrix))
