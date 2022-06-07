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
from anti_clustering import ExchangeHeuristicAntiClustering, SimulatedAnnealingHeuristicAntiClustering, \
    NaiveRandomHeuristicAntiClustering, ExactClusterEditingAntiClustering, AntiClustering

from sklearn import datasets
import pandas as pd

iris_data = datasets.load_iris(as_frame=True)
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

methods: List[AntiClustering] = [
    ExchangeHeuristicAntiClustering(), SimulatedAnnealingHeuristicAntiClustering(alpha=0.9, iterations=2000),
    NaiveRandomHeuristicAntiClustering(), ExactClusterEditingAntiClustering()
]

for method in methods:
    for k in range(2, 4):
        print(f"Method: {method.__class__.__name__}, clusters: {k}")

        start_time = time.time()
        df = method.run(
            df=iris_df,
            numeric_columns=list(iris_df.columns),
            categorical_columns=None,
            num_groups=k,
            destination_column='Cluster'
        )
        time_taken = time.time() - start_time

        # Mean and stddev for each cluster for each feature
        aggregated_df = df.groupby('Cluster').agg(['mean', 'std'])
        # Absolute difference between min and max mean/stddev in each feature
        difference_df = aggregated_df.max() - aggregated_df.min()
        # Mean of differences
        mean_df = difference_df.reset_index(level=[1]).groupby(['level_1']).mean()

        print(f"∆M: {mean_df.loc['mean'][0]}")
        print(f"∆SD: {mean_df.loc['std'][0]}")
        print(f"Running time: {time_taken}s")
