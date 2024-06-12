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
A tabu search with restarts approach to solving the anti-clustering problem.
"""

import numpy.typing as npt
from anti_clustering._cluster_swap_heuristic import ClusterSwapHeuristic


class TabuSearchHeuristicAntiClustering(ClusterSwapHeuristic):
    """
    A tabu search with restarts approach to solving the anti-clustering problem.
    In this version, specific transformations are put in the tabu list not solutions.
    """

    def __init__(
        self,
        verbose: bool = False,
        random_seed: int = None,
        tabu_tenure: int = 10,
        iterations: int = 2000,
        restarts: int = 9,
    ):
        # pylint: disable = R0913
        super().__init__(verbose=verbose, random_seed=random_seed)
        self.tabu_tenure = tabu_tenure
        self.iterations = iterations
        self.restarts = restarts

    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        # Start with random cluster assignment
        cluster_assignment = self._get_random_clusters(num_groups=num_groups, num_elements=len(distance_matrix))

        if self.verbose:
            print("Solving")

        candidate_solutions = []

        for restart in range(self.restarts):
            tabu_swaps = []
            # Initial objective value
            objective = self._calculate_objective(cluster_assignment, distance_matrix)
            for iteration in range(self.iterations):
                if self.verbose and iteration % 5 == 0:
                    print(f"Iteration {iteration + 1} of {self.iterations}")

                # Select random element
                i = self.rnd.randint(0, len(distance_matrix) - 1)

                # Get possible swaps
                possible_exchanges = [
                    j
                    for j in self._get_exchanges(cluster_assignment, i)
                    if (i, j) not in tabu_swaps and (j, i) not in tabu_swaps
                ]

                if len(possible_exchanges) == 0:
                    continue

                # Generate possible assignments
                j = possible_exchanges[self.rnd.randint(0, len(possible_exchanges) - 1)]

                # Select random possible swap.
                new_cluster_assignment = self._swap(cluster_assignment, i, j)
                new_objective = self._calculate_objective(new_cluster_assignment, distance_matrix)

                # Select solution as current if it improves the objective value
                if new_objective > objective:
                    cluster_assignment = new_cluster_assignment
                    objective = new_objective
                    tabu_swaps.append((i, j))
                    # Delete oldest tabu swap if tabu list is full
                    if len(tabu_swaps) > self.tabu_tenure:
                        tabu_swaps.pop(0)

            candidate_solutions.append((objective, cluster_assignment))

            if self.verbose:
                print(f"Restart {restart + 1} of {self.restarts}")

            # Cold restart, select random cluster assignment
            cluster_assignment = self._get_random_clusters(num_groups=num_groups, num_elements=len(distance_matrix))

        # Select best solution, maximizing objective
        _, best_cluster_assignment = max(candidate_solutions, key=lambda x: x[0])

        return best_cluster_assignment
