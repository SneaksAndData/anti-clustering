from typing import List
from abc import ABC, abstractmethod
import pandas as pd


class AntiClustering(ABC):
    @abstractmethod
    def run(self, df: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str], num_groups: int, destination_column: str)  -> pd.DataFrame:
        pass
