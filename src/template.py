import numpy as np

from abc import ABC, abstractmethod
from typing import Literal, Tuple


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
    def fit(self, train: Tuple[np.ndarray, np.ndarray], val: Tuple[np.ndarray, np.ndarray]) -> None:
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        ...
