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
Example of how to evaluate results.

Based on:
Papenberg, M., & Klau, G. W. (2021). Using anticlustering to partition data sets into equivalent parts.
Psychological Methods, 26(2), 161–174. https://doi.org/10.1037/met0000301
"""

import time
from typing import List
from anti_clustering import (
    ExchangeHeuristicAntiClustering,
    SimulatedAnnealingHeuristicAntiClustering,
    NaiveRandomHeuristicAntiClustering,
    TabuSearchHeuristicAntiClustering,
    ExactClusterEditingAntiClustering,
    AntiClustering,
)

from sklearn import datasets
import pandas as pd

iris_data = datasets.load_iris(as_frame=True)
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

methods: List[AntiClustering] = [
    TabuSearchHeuristicAntiClustering(iterations=5000, restarts=10, tabu_tenure=50),
    ExchangeHeuristicAntiClustering(),
    SimulatedAnnealingHeuristicAntiClustering(alpha=0.95, iterations=5000, starting_temperature=1000, restarts=15),
    NaiveRandomHeuristicAntiClustering(),
    # ExactClusterEditingAntiClustering(), # This method is extremely slow for large datasets
]

for k in range(2, 4):
    print(f"------------- Number of clusters: {k} -------------")
    summary = []
    for method in methods:
        print(f"Running method: {method.__class__.__name__}")

        start_time = time.time()
        df = method.run(
            df=iris_df,
            numerical_columns=list(iris_df.columns),
            categorical_columns=None,
            num_groups=k,
            destination_column="Cluster",
        )
        time_taken = time.time() - start_time

        # Mean and stddev for each cluster for each feature
        aggregated_df = df.groupby("Cluster").agg(["mean", "std"])
        # Absolute difference between min and max mean/stddev in each feature
        difference_df = aggregated_df.max() - aggregated_df.min()
        # Mean of differences
        mean_df = difference_df.reset_index(level=[1]).groupby(["level_1"]).mean()

        summary.append(
            pd.DataFrame(
                {
                    "Method": [method.__class__.__name__],
                    "Clusters": [k],
                    "∆M": [round(mean_df.loc["mean"][0], 4)],
                    "∆SD": [round(mean_df.loc["std"][0], 4)],
                    "Time (s)": [time_taken],
                }
            )
        )
    print("Summary (lower ∆M and ∆SD is better):")
    print(pd.concat(summary).to_string())
