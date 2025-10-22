import pandas as pd
import numpy as np

from omegaconf import OmegaConf
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.template import Algorithm

class CustomXGBClassifier(Algorithm):
    def __init__(self,
        param_grid: dict[str, list] = None,
        use_oob: bool = False,
    ):
        super().__init__(
            task_type="classification",
            param_grid=param_grid,
        )
        self.model = None
        self.use_oob = use_oob
        self.param_grid = OmegaConf.to_container(self.param_grid, resolve=True)

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
            XGBClassifier(
                random_state=seed,
            ),
            self.param_grid,
            cv=[(train_idx, val_idx)],
            n_jobs=-1,
        )
        self.model.fit(X_all, y_all)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)