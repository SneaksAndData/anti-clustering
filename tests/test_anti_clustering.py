from anti_clustering import (
    SimulatedAnnealingHeuristicAntiClustering,
    ExactClusterEditingAntiClustering,
    ExchangeHeuristicAntiClustering,
    NaiveRandomHeuristicAntiClustering,
)
import pytest
import pandas as pd


@pytest.mark.parametrize(
    "df, optimal_clusters, num_groups",
    [
        (pd.DataFrame(data={"x": [0, 0, 2, 3, 3, 2], "y": [1, 2, 2, 1, 0, 0]}), [0, 0, 0, 0, 0, 0], 1),
        (pd.DataFrame(data={"x": [0, 0, 2, 3, 3, 2], "y": [1, 2, 2, 1, 0, 0]}), [0, 1, 0, 1, 0, 1], 2),
        (pd.DataFrame(data={"x": [0, 0, 2, 3, 3, 2], "y": [1, 2, 2, 1, 0, 0]}), [0, 1, 2, 0, 1, 2], 3),
        (pd.DataFrame(data={"x": [0, 2, 3, 0, 3, 2], "y": [1, 2, 0, 2, 1, 0]}), [0, 2, 1, 1, 0, 2], 3),
        (pd.DataFrame(data={"x": [0, 2, 3, 0, 3, 2], "y": [1, 2, 0, 2, 1, 0]}), [0, 0, 0, 1, 1, 1], 2),
        (pd.DataFrame(data={"x": [0, 2, 3, 0, 3, 2], "y": [1, 2, 0, 2, 1, 0]}), [0, 0, 0, 0, 0, 0], 1),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        ExchangeHeuristicAntiClustering(random_seed=1),
        SimulatedAnnealingHeuristicAntiClustering(random_seed=1),
        ExactClusterEditingAntiClustering(),
        NaiveRandomHeuristicAntiClustering(random_seed=1),
    ],
)
def test_optimal_numerical_anti_clustering(df, optimal_clusters, num_groups, algorithm):
    """
    Test that numerical anti clustering returns optimal result except for NaiveRandomHeuristicAntiClustering.
    """
    column = "Cluster"
    result_df = algorithm.run(
        df=df, numerical_columns=["x", "y"], num_groups=num_groups, destination_column=column, categorical_columns=None
    )

    # NaiveRandomHeuristicAntiClustering will not find the optimal solution
    if not isinstance(algorithm, NaiveRandomHeuristicAntiClustering):
        # Assert optimal solutions is found
        assert (optimal_clusters == result_df[column].to_numpy()).all()

    # Assert that num_groups clusters are generated
    assert result_df[column].nunique() == num_groups


@pytest.mark.parametrize(
    "df, optimal_clusters, num_groups",
    [
        (pd.DataFrame(data={"c": ["a", "b", "c", "a", "b", "c"]}), [0, 0, 1, 1, 1, 0], 2),
        (pd.DataFrame(data={"c": ["a", "b", "a", "b", "a", "b"]}), [0, 0, 1, 1, 2, 2], 3),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        ExchangeHeuristicAntiClustering(random_seed=1),
        SimulatedAnnealingHeuristicAntiClustering(random_seed=1),
        ExactClusterEditingAntiClustering(),
        NaiveRandomHeuristicAntiClustering(random_seed=1),
    ],
)
def test_optimal_categorical_anti_clustering(df, optimal_clusters, num_groups, algorithm):
    """
    Test that categorical anti clustering returns optimal result except for NaiveRandomHeuristicAntiClustering.
    """
    column = "Cluster"
    result_df = algorithm.run(
        df=df, numerical_columns=None, num_groups=num_groups, destination_column=column, categorical_columns=["c"]
    )

    # Assert that num_groups clusters are generated
    assert result_df[column].nunique() == num_groups

    # NaiveRandomHeuristicAntiClustering will not find the optimal solution
    if not isinstance(algorithm, NaiveRandomHeuristicAntiClustering):
        # Assert all categories are contained in all clusters
        assert (result_df.groupby(column)["c"].nunique() == result_df["c"].nunique()).all()


@pytest.mark.parametrize(
    "algorithm",
    [
        ExchangeHeuristicAntiClustering(random_seed=1, verbose=True),
        SimulatedAnnealingHeuristicAntiClustering(random_seed=1, verbose=True),
        ExactClusterEditingAntiClustering(verbose=True),
        NaiveRandomHeuristicAntiClustering(random_seed=1, verbose=True),
    ],
)
def test_verbose_mode(algorithm):
    """
    Test that verbose mode does not raise exception.
    """
    df = pd.DataFrame(data={"x": [0, 0, 2, 3, 3, 2], "y": [1, 2, 2, 1, 0, 0], "c": ["cat1"] * 6})
    algorithm.run(
        df=df, numerical_columns=["x", "y"], num_groups=2, destination_column="col", categorical_columns=["c"]
    )
