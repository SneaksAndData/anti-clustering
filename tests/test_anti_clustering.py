from anti_clustering.exact_cluster_editing import ExactClusterEditingAntiClustering
from anti_clustering.exchange_heuristic import ExchangeHeuristicAntiClustering
from anti_clustering.simulated_annealing_heuristic import SimulatedAnnealingHeuristicAntiClustering
import pytest
import pandas as pd


@pytest.mark.parametrize(
    "df, labels, num_groups",
    [
        (
            pd.DataFrame(data={
                'x': [0, 0, 2, 3, 3, 2],
                'y': [1, 2, 2, 1, 0, 0],
                'c': ['cat1'] * 6
            }),
            [0, 0, 0, 0, 0, 0],
            1
        ),
        (
            pd.DataFrame(data={
                'x': [0, 0, 2, 3, 3, 2],
                'y': [1, 2, 2, 1, 0, 0],
                'c': ['cat1'] * 6
            }),
            [0, 1, 0, 1, 0, 1],
            2
        ),
        (
            pd.DataFrame(data={
                'x': [0, 0, 2, 3, 3, 2],
                'y': [1, 2, 2, 1, 0, 0],
                'c': ['cat1'] * 6
            }),
            [0, 1, 2, 0, 1, 2],
            3
        ),
        (
            pd.DataFrame(data={
                'x': [0, 2, 3, 0, 3, 2],
                'y': [1, 2, 0, 2, 1, 0],
                'c': ['cat1'] * 6
            }),
            [0, 2, 1, 1, 0, 2],
            3
        ),
        (
            pd.DataFrame(data={
                'x': [0, 2, 3, 0, 3, 2],
                'y': [1, 2, 0, 2, 1, 0],
                'c': ['cat1'] * 6
            }),
            [0, 0, 0, 1, 1, 1],
            2
        ),
        (
            pd.DataFrame(data={
                'x': [0, 2, 3, 0, 3, 2],
                'y': [1, 2, 0, 2, 1, 0],
                'c': ['cat1'] * 6
            }),
            [0, 0, 0, 0, 0, 0],
            1
        ),
        # (
        #     pd.DataFrame(data={
        #         'x': [0, 0, 0, 0, 0, 0],
        #         'y': [1, 1, 1, 1, 1, 1],
        #         'c': ['a', 'b', 'c', 'a', 'b', 'c']
        #     }),
        #     [0, 0, 2, 2, 2, 0],
        #     2
        # ),
        # (
        #     pd.DataFrame(data={
        #         'x': [0, 0, 0, 0, 0, 0],
        #         'y': [1, 1, 1, 1, 1, 1],
        #         'c': ['a', 'b', 'a', 'b', 'a', 'b']
        #     }),
        #     [0, 0, 2, 2, 4, 4],
        #     3
        # )
    ]
)
@pytest.mark.parametrize("algorithm", [
    ExchangeHeuristicAntiClustering(random_seed=1),
    SimulatedAnnealingHeuristicAntiClustering(random_seed=1),
    ExactClusterEditingAntiClustering(),
])
def test_ilp_anti_clustering(df, labels, num_groups, algorithm):
    column = 'Cluster'
    result_df = algorithm.run(df=df, numeric_columns=['x', 'y'], num_groups=num_groups, destination_column=column, categorical_columns=['c'])
    assert (labels == result_df[column].to_numpy()).all()
    assert result_df[column].nunique() == num_groups
