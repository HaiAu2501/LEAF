import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from abc import ABC, abstractmethod
from typing import Literal
from utils.logger import Logger

FeatureMatrix = pd.DataFrame | np.ndarray
TargetMatrix = pd.Series | np.ndarray
DataTuple = tuple[FeatureMatrix, TargetMatrix]

OmegaConf.register_new_resolver(
    "range",
    lambda start, end: list(range(int(start), int(end)))
)
OmegaConf.register_new_resolver(
    "logspace", 
    lambda start, end, num: np.logspace(float(start), float(end), int(num)).tolist()
)
OmegaConf.register_new_resolver(
    "linspace", 
    lambda start, end, num: np.linspace(float(start), float(end), int(num)).tolist()
)

class Algorithm(ABC):
    def __init__(
        self, 
        logger: Logger,
        name: str,
        task_type: Literal["classification", "regression"],
        param_grid: dict[str, list] = None
    ):
        self.task_type = task_type
        if param_grid is None:
            param_grid = {}
        self.param_grid = OmegaConf.to_container(param_grid, resolve=True)
        self.logger = logger
        logger.log_to_json(
            {"name": name, "task_type": task_type, "param_grid": self.param_grid},
            f"{name}_config.json"
        )

    @abstractmethod
    def fit(self, train: DataTuple, val: DataTuple, seed: int) -> None:
        ...

    @abstractmethod
    def predict(self, X_test: FeatureMatrix) -> FeatureMatrix:
        ...
