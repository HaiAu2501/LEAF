import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from src.template import Algorithm
from utils.logger import Logger

class CustomRFClassifier(Algorithm):
    """
    Custom Random Forest Classifier with optional hyperparameter tuning using GridSearchCV.
    """
    def __init__(
        self,
        logger: Logger,
        param_grid: dict[str, list] = None,
    ):
        super().__init__(
            logger=logger,
            name="RandomForest",
            task_type="classification",
            param_grid=param_grid,
        )
        self.model = None

    def fit(self, train: tuple, val: tuple, seed: int) -> None:
        self.model = None
        X_train, y_train = train
        X_val, y_val = val
        X_all = pd.concat([X_train, X_val], ignore_index=True)
        y_all = np.concatenate([y_train, y_val])

        n_train = len(X_train)
        n_val = len(X_val)

        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_train + n_val)

        self.model = GridSearchCV(
            RandomForestClassifier(
                random_state=seed,
                class_weight="balanced"
            ),
            self.param_grid,
            cv=[(train_idx, val_idx)],
            n_jobs=-1,
        )
        self.model.fit(X_all, y_all)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)