import numpy as np

from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from src.template import Algorithm

class CustomRFClassifier(Algorithm):
    """
    Custom Random Forest Classifier with optional hyperparameter tuning using GridSearchCV.
    """

    def __init__(self,
        param_grid: dict[str, list] = None,
        use_oob: bool = False,
    ):
        super().__init__(
            task_type="classification",
            param_grid=param_grid,
        )
        self.use_oob = use_oob
        self.param_grid = OmegaConf.to_container(self.param_grid, resolve=True)

    def fit(self, train: tuple, val: tuple, seed: int) -> None:
        X_train, y_train = train
        if self.param_grid and self.use_oob:
            # Hyperparameter tuning with OOB score
            best_params, best_score = None, -np.inf
            for params in ParameterGrid(self.param_grid):
                est = RandomForestClassifier(
                    oob_score=True,
                    random_state=seed,
                    **params
                )
                est.fit(X_train, y_train)
                if est.oob_score_ > best_score:
                    best_score, best_params = est.oob_score_, params

            # Refit the model with the best parameters
            self.model = RandomForestClassifier(
                oob_score=True,
                random_state=seed,
                **best_params
            )
            self.model.fit(X_train, y_train)
        elif self.param_grid:
            # Classic GridSearchCV (without OOB)
            X_val, y_val = val
            X_all = np.concatenate([X_train, X_val])
            y_all = np.concatenate([y_train, y_val])

            n_train = len(X_train)
            n_val = len(X_val)

            train_idx = np.arange(n_train)
            val_idx = np.arange(n_train, n_train + n_val)

            self.model = GridSearchCV(
                RandomForestClassifier(random_state=seed),
                self.param_grid,
                cv=[(train_idx, val_idx)],
                n_jobs=-1,
                refit=True
            )
            self.model.fit(X_all, y_all)

    def predict(self, test) -> np.ndarray:
        return self.model.predict(test)