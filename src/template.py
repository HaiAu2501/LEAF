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
        self.name = name
        if param_grid is None:
            param_grid = {}
        else:
            param_grid = OmegaConf.to_container(param_grid, resolve=True)
        self.param_grid: dict[str, list] = param_grid
        self.logger = logger
        logger.log_to_json(
            {"name": self.name, "task_type": self.task_type, "param_grid": self.param_grid},
            f"{self.name}_config.json"
        )
        self.prior_constructor = None

    def setup(self, dataset, model):
        print("[WARNING] [Algorithm] Your algorithm does not use prior information.")
        pass

    @abstractmethod
    def fit(self, train: DataTuple, val: DataTuple, seed: int) -> None:
        ...

    @abstractmethod
    def predict(self, X_test: FeatureMatrix) -> FeatureMatrix:
        ...
