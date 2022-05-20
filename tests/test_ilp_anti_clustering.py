from anti_clustering.ilp import ILPAntiClustering
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
            [0, 1, 2, 2, 0, 1],
            3
        ),
        (
            pd.DataFrame(data={
                'x': [0, 2, 3, 0, 3, 2],
                'y': [1, 2, 0, 2, 1, 0],
                'c': ['cat1'] * 6
            }),
            [0, 0, 0, 3, 3, 3],
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
        (
            pd.DataFrame(data={
                'x': [0, 0, 0, 0, 0, 0],
                'y': [1, 1, 1, 1, 1, 1],
                'c': ['a', 'b', 'c', 'a', 'b', 'c']
            }),
            [0, 0, 2, 2, 2, 0],
            2
        ),
        (
            pd.DataFrame(data={
                'x': [0, 0, 0, 0, 0, 0],
                'y': [1, 1, 1, 1, 1, 1],
                'c': ['a', 'b', 'a', 'b', 'a', 'b']
            }),
            [0, 0, 2, 2, 4, 4],
            3
        )
    ]
)
def test_ilp_anti_clustering(df, labels, num_groups):
    algorithm = ILPAntiClustering()
    column = 'Cluster'
    result_df = algorithm.run(df=df, numeric_columns=['x', 'y'], num_groups=num_groups, destination_column=column, categorical_columns=['c'])
    assert result_df[column].nunique() == num_groups
    assert (labels == result_df[column].to_numpy()).all()
