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
A simulated annealing with restarts approach to solving the anti-clustering problem.
"""

import math
import numpy as np
import numpy.typing as npt
from anti_clustering._cluster_swap_heuristic import ClusterSwapHeuristic


class SimulatedAnnealingHeuristicAntiClustering(ClusterSwapHeuristic):
    """
    A simulated annealing with restarts approach to solving the anti-clustering problem.
    """

    def __init__(
        self,
        verbose: bool = False,
        random_seed: int = None,
        alpha: float = 0.9,
        iterations: int = 2000,
        starting_temperature: float = 100,
        restarts: int = 9,
    ):
        # pylint: disable = R0913
        super().__init__(verbose=verbose, random_seed=random_seed)
        self.alpha = alpha
        self.iterations = iterations
        self.starting_temperature = starting_temperature
        self.restarts = restarts

    def _solve(self, distance_matrix: npt.NDArray[float], num_groups: int) -> npt.NDArray[bool]:
        # Start with random cluster assignment
        cluster_assignment = self._get_random_clusters(num_groups=num_groups, num_elements=len(distance_matrix))

        if self.verbose:
            print("Solving")

        candidate_solutions = []

        for restart in range(self.restarts):
            temperature = self.starting_temperature
            # Initial objective value
            objective = self._calculate_objective(cluster_assignment, distance_matrix)
            for iteration in range(self.iterations):
                if self.verbose and iteration % 5 == 0:
                    print(f"Iteration {iteration + 1} of {self.iterations}")

                # Select random element
                i = self.rnd.randint(0, len(distance_matrix) - 1)
                # Get possible swaps
                possible_exchanges = self._get_exchanges(cluster_assignment, i)
                if len(possible_exchanges) == 0:
                    continue
                # Select random possible swap.
                j = possible_exchanges[self.rnd.randint(0, len(possible_exchanges) - 1)]

                new_cluster_assignment = self._swap(cluster_assignment, i, j)
                new_objective = self._calculate_objective(new_cluster_assignment, distance_matrix)

                # Select solution as current if accepted
                if self._accept(new_objective - objective, temperature):
                    objective = new_objective
                    cluster_assignment = new_cluster_assignment

                # Cool down temperature
                temperature = temperature * self.alpha

            candidate_solutions.append((objective, cluster_assignment))

            if self.verbose:
                print(f"Restart {restart + 1} of {self.restarts}")

            # Cold restart, select random cluster assignment
            cluster_assignment = self._get_random_clusters(num_groups=num_groups, num_elements=len(distance_matrix))

        # Select best solution, maximizing objective
        _, best_cluster_assignment = max(candidate_solutions, key=lambda x: x[0])

        return best_cluster_assignment

    def _calculate_objective(self, cluster_assignment: npt.NDArray[bool], distance_matrix: npt.NDArray[float]) -> float:
        """
        Calculate objective value
        :param cluster_assignment: Cluster assignment
        :param distance_matrix: Distance matrix
        :return: Objective value
        """
        return np.multiply(cluster_assignment, distance_matrix).sum()

    def _accept(self, delta: float, temperature: float) -> bool:
        """
        Simulated annealing acceptance function. Notice d/t is used instead of -d/t because we are maximizing.
        :param delta: Difference in objective
        :param temperature: Current temperature
        :return: Whether the solution is accepted or not.
        """
        return delta >= 0 or math.exp(delta / temperature) >= self.rnd.uniform(0, 1)
