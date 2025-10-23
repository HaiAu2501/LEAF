import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Literal

FeatureMatrix = pd.DataFrame | np.ndarray
TargetMatrix = pd.Series | np.ndarray
DataTuple = tuple[FeatureMatrix, TargetMatrix]

class Algorithm(ABC):
    def __init__(
        self, 
        task_type: Literal["classification", "regression"],
        param_grid: dict[str, list] = None,
    ):
        self.task_type = task_type
        if param_grid is None:
            param_grid = {}
        self.param_grid = param_grid

    @abstractmethod
    def fit(self, train: DataTuple, val: DataTuple, seed: int) -> None:
        ...

    @abstractmethod
    def predict(self, X_test: FeatureMatrix) -> FeatureMatrix:
        ...
