# Anti-clustering

A generic Python library for solving the anti-clustering problem. While clustering algorithms will achieve high similarity within a cluster and low similarity between clusters, the anti-clustering algorithms will achieve the opposite; namely to minimise similarity within a cluster and maximise the similarity between clusters.
Currently, a handful of algorithms are implemented in this library:
* An exact approach using a BIP formulation.
* An enumerated exchange heuristic.
* A simulated annealing heuristic.

Keep in mind anti-clustering is computationally difficult problem and may run slow even for small instance sizes. The current ILP does not finish in reasonable time when anti-clustering the Iris dataset (150 data points).

The two former approaches are implemented as describing in following paper:\
*Papenberg, M., & Klau, G. W. (2021). Using anticlustering to partition data sets into equivalent parts.
Psychological Methods, 26(2), 161–174. [DOI](https://doi.org/10.1037/met0000301). [Preprint](https://psyarxiv.com/3razc/)* \
The paper is accompanied by a library for the R programming language: [anticlust](https://github.com/m-Py/anticlust).

Differently to the [anticlust](https://github.com/m-Py/anticlust) R package, this library currently only have one objective function. 
In this library the objective will maximise intra-cluster distance: Euclidean distance for numerical columns and Hamming distance for categorical columns.

## Use cases
Within software testing, anti-clustering can be used for generating test and control groups in AB-testing.
Example: You have a webshop with a number of users. The webshop is undergoing active development and you have a new feature coming up. 
This feature should be tested against as many different users as possible without testing against the entire user-base. 
For that you can create a maximally diverse subset of the user-base to test against (the A group). 
The remaining users (B group) will not test this feature. For dividing the user-base you can use the anti-clustering algorithms. 
A and B groups should be as similar as possible to have a reliable basis of comparison, but internally in group A (and B) the elements should be as dissimilar as possible.

This is just one use case, probably many more exists.

## Usage
The input to the algorithm is a Pandas dataframe with each row representing a data point. The output is the same dataframe with an extra column containing integer encoded cluster labels. Below is an example based on the Iris dataset:
```python
from anti_clustering import ExactClusterEditingAntiClustering
from sklearn import datasets
import pandas as pd

iris_data = datasets.load_iris(as_frame=True)
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

algorithm = ExactClusterEditingAntiClustering()

df = algorithm.run(
    df=iris_df,
    numerical_columns=list(iris_df.columns),
    categorical_columns=None,
    num_groups=2,
    destination_column='Cluster'
)
```

## Contributions
If you have any suggestions or have found a bug, feel free to open issues. If you have implemented a new algorithm or know how to tweak the existing ones; PRs are very appreciated.

## License
This library is licensed under the Apache 2.0 license.
