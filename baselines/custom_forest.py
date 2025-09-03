from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List, Union
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


class CustomForest(BaseEstimator):
    """
    A reproducible bootstrap forest of Decision Trees with per-tree GridSearch.

    Parameters
    ----------
    task_type : {"regression", "classification"}
        Loại bài toán.
    n_estimators : int
        Số cây trong rừng.
    param_grid : dict or None
        Lưới tham số cho DecisionTree*; được áp dụng bằng GridSearch **riêng từng cây**.
        Nếu None hoặc rỗng, mỗi cây sẽ fit trực tiếp không GridSearch.
    cv : int or CV splitter or None
        - Nếu int: với classification dùng StratifiedKFold(cv, shuffle=True, random_state=seed_i),
          regression dùng KFold(cv, shuffle=True, random_state=seed_i).
        - Nếu cung cấp splitter, sẽ dùng nguyên splitter đó (chịu trách nhiệm về reproducibility).
        - Nếu None: mặc định là 3.
    scoring : str or None
        Scoring cho GridSearch. Nếu None, mặc định:
        - classification: "balanced_accuracy"
        - regression: "neg_mean_squared_error"
    bootstrap : bool
        Có bootstrap mẫu cho mỗi cây không (with replacement).
    max_samples : None | int | float
        - None: lấy đúng N mẫu cho bootstrap.
        - int: số mẫu bootstrap.
        - float in (0,1]: tỉ lệ mẫu trên N.
        Với bootstrap=False: nếu max_samples None → dùng toàn bộ; nếu đặt giá trị → sample không lặp.
    random_state : int or None
        Hạt giống của rừng. Cây thứ i dùng seed = random_state + i (i bắt đầu từ 1).
    n_jobs : int or None
        n_jobs cho GridSearchCV (không song song giữa các cây).
    verbose : int
        Verbosity cho GridSearchCV.
    base_tree_params : dict
        Tham số mặc định cho DecisionTreeClassifier/Regressor (ví dụ: max_depth, min_samples_split,...).

    Attributes
    ----------
    estimators_ : List[DecisionTree*]
        Danh sách các cây đã fit.
    seeds_ : List[int]
        Seed dùng cho từng cây.
    bootstrap_indices_ : List[np.ndarray]
        Chỉ số mẫu (có thứ tự) đã chọn cho bootstrap của từng cây.
    classes_ : np.ndarray (chỉ với classification)
        Tập nhãn toàn cục, để căn chỉnh xác suất khi một số cây không thấy đủ lớp trên bootstrap.
    """

    def __init__(
        self,
        task_type: Literal["regression", "classification"],
        n_estimators: int = 100,
        param_grid: Optional[Dict[str, Any]] = None,
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        base_tree_params: Optional[Dict[str, Any]] = None,
    ):
        self.task_type = task_type
        self.n_estimators = int(n_estimators)
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.base_tree_params = base_tree_params or {}

        # Fitted attributes
        self.estimators_: List[Any] = []
        self.seeds_: List[int] = []
        self.bootstrap_indices_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None

    # ----------- helpers -----------
    def _rng_for_tree(self, i: int) -> np.random.RandomState:
        # i starts at 1 for readability; if random_state is None -> nondeterministic
        seed = None if self.random_state is None else int(self.random_state) + i
        return np.random.RandomState(seed)

    def _compute_n_bootstrap(self, n_samples: int) -> int:
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, int):
            return int(self.max_samples)
        if isinstance(self.max_samples, float):
            if not (0.0 < self.max_samples <= 1.0):
                raise ValueError("max_samples as float must be in (0, 1].")
            return max(1, int(np.floor(self.max_samples * n_samples)))
        raise ValueError("max_samples must be None, int, or float in (0,1].")

    def _bootstrap_indices(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        if self.bootstrap:
            size = self._compute_n_bootstrap(n_samples)
            # with replacement, order preserved by randint -> reproducible sequence
            return rng.randint(0, n_samples, size=size)
        else:
            if self.max_samples is None:
                return np.arange(n_samples, dtype=int)
            size = self._compute_n_bootstrap(n_samples)
            # without replacement, but deterministic due to rng
            return rng.choice(n_samples, size=size, replace=False)

    def _make_cv(self, seed_i: Optional[int], y: Optional[np.ndarray]) -> Any:
        if isinstance(self.cv, int) or self.cv is None:
            n_splits = int(self.cv) if isinstance(self.cv, int) else 3
            if self.task_type == "classification" and y is not None and type_of_target(y) in ("binary", "multiclass"):
                return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_i)
            else:
                return KFold(n_splits=n_splits, shuffle=True, random_state=seed_i)
        # Provided splitter (assumed deterministic by caller)
        return self.cv

    def _default_scoring(self) -> str:
        if self.scoring is not None:
            return self.scoring
        return "balanced_accuracy" if self.task_type == "classification" else "neg_mean_squared_error"

    def _base_estimator(self, seed_i: Optional[int]):
        if self.task_type == "classification":
            return DecisionTreeClassifier(random_state=seed_i, **self.base_tree_params)
        else:
            return DecisionTreeRegressor(random_state=seed_i, **self.base_tree_params)

    # ----------- sklearn API -----------
    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        # classes_ for classification (to unify predict_proba shapes across trees)
        if self.task_type == "classification":
            self.classes_ = np.unique(y)

        self.estimators_.clear()
        self.seeds_.clear()
        self.bootstrap_indices_.clear()

        for i in range(1, self.n_estimators + 1):
            rng = self._rng_for_tree(i)
            # extract seed value for record (None if no random_state)
            seed_i = None if self.random_state is None else int(self.random_state) + i
            idx = self._bootstrap_indices(n_samples, rng)
            X_boot, y_boot = X[idx], y[idx]
            sw_boot = sample_weight[idx] if sample_weight is not None else None

            est = self._base_estimator(seed_i)
            # Per-tree CV (deterministic w.r.t seed_i when cv is int/None)
            cv_i = self._make_cv(seed_i, y_boot if self.task_type == "classification" else None)

            if self.param_grid:
                # GridSearch per tree on its bootstrap sample
                gs = GridSearchCV(
                    estimator=est,
                    param_grid=self.param_grid,
                    cv=cv_i,
                    scoring=self._default_scoring(),
                    n_jobs=self.n_jobs,
                    refit=True,
                    verbose=self.verbose,
                )
                if sw_boot is not None:
                    gs.fit(X_boot, y_boot, **{"sample_weight": sw_boot})
                else:
                    gs.fit(X_boot, y_boot)
                best_est = gs.best_estimator_
            else:
                if sw_boot is not None:
                    est.fit(X_boot, y_boot, sample_weight=sw_boot)
                else:
                    est.fit(X_boot, y_boot)
                best_est = est

            self.estimators_.append(best_est)
            self.seeds_.append(seed_i)
            self.bootstrap_indices_.append(idx)

        return self

    def _predict_proba_aligned(self, est, X) -> np.ndarray:
        """Align per-tree proba to global classes_ (handles missing classes in bootstrap)."""
        proba = est.predict_proba(X)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        est_classes = est.classes_
        if np.array_equal(est_classes, self.classes_):
            return proba
        # map columns
        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        # build index map from estimator's classes to global classes
        class_to_pos = {c: j for j, c in enumerate(self.classes_)}
        cols = [class_to_pos[c] for c in est_classes]
        aligned[:, cols] = proba
        return aligned

    def predict(self, X):
        X = np.asarray(X)
        if self.task_type == "regression":
            preds = np.column_stack([est.predict(X) for est in self.estimators_])
            return np.mean(preds, axis=1)
        else:
            # majority vote via averaged proba (tie-break by argmax)
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        X = np.asarray(X)
        # average probabilities across trees (aligned to global classes)
        probas = np.array([self._predict_proba_aligned(est, X) for est in self.estimators_])
        return np.mean(probas, axis=0)

    def __len__(self):
        return len(self.estimators_)
