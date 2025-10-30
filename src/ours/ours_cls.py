import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.template import Algorithm
from src.ours.prior_constructor import PriorConstructor
from utils.logger import Logger


class OursClassifier(Algorithm):
    def __init__(
        self,
        eta: list[float],
        logger: Logger,
        param_grid: dict[str, list] = None,
    ):
        super().__init__(
            logger=logger,
            name="OursClassifier",
            task_type="classification",
            param_grid=param_grid,
        )
        self.prior_constructor = None
        self.eta_grid = list(eta)
        self.trees: list[DecisionTreeClassifier] = []
        self.logPs: list[float] = []
        self.losses_val: list[float] = []
        self.q: np.ndarray | None = None
        self.best_eta: float | None = None
        self.classes_: np.ndarray | None = None
        self.rng: np.random.Generator | None = None
    
    def setup(self, dataset, model):
        self.prior_constructor = PriorConstructor(
            dataset=dataset,
            model=model,
            logger=self.logger
        )
        pass

    def fit(self, train, val, seed):
        # Reset state
        self.trees = []
        self.logPs = []
        self.losses_val = []
        self.q = None
        self.best_eta = None
        self.rng = np.random.default_rng(seed)

        X_train, y_train = train
        X_val, y_val = val
        assert isinstance(X_train, pd.DataFrame) and isinstance(X_val, pd.DataFrame), "Expect DataFrame inputs"
        self.classes_ = np.unique(y_train)

        # Determine number of trees
        n_estimators_list = self.param_grid.get("n_estimators", [50])
        n_trees = int(max(n_estimators_list)) if len(n_estimators_list) > 0 else 50

        # Build candidate hyperparam combos excluding n_estimators
        grid_no_n = {k: v for k, v in self.param_grid.items() if k != "n_estimators"}
        if len(grid_no_n) == 0:
            combos = [dict()]
        else:
            keys = list(grid_no_n.keys())
            vals = [grid_no_n[k] for k in keys]
            combos = [dict(zip(keys, prod)) for prod in itertools.product(*vals)]

        # Feature weights and names
        fw = self.prior_constructor.feature_weights
        feat_names = list(X_train.columns)
        # Sanity: ensure coverage
        if set(fw.keys()) != set(feat_names):
            raise ValueError("Feature weight coverage mismatch with training features.")
        w = np.array([fw[f] for f in feat_names], dtype=float)
        w = np.clip(w, 0.0, None)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("Invalid feature weights.")
        p_feat = w / w.sum()

        d = len(feat_names)
        m_sub = max(1, int(np.ceil(np.sqrt(d))))  # subset size per tree

        N = len(X_train)

        for i in range(n_trees):
            # 1) Sample feature subset according to priors
            subset_idx = self.rng.choice(d, size=m_sub, replace=False, p=p_feat)
            subset = [feat_names[j] for j in subset_idx]

            # 2) Bootstrap sample indices
            # Ensure OOB non-empty (retry up to 5 times)
            for attempt in range(5):
                boot_idx = self.rng.integers(0, N, size=N)
                in_boot = np.zeros(N, dtype=bool)
                in_boot[boot_idx] = True
                oob_idx = np.where(~in_boot)[0]
                if oob_idx.size > 0:
                    break
            if oob_idx.size == 0:
                # fallback: make 20% OOB deterministically
                oob_mask = np.zeros(N, dtype=bool)
                oob_mask[self.rng.choice(N, size=max(1, N // 5), replace=False)] = True
                oob_idx = np.where(oob_mask)[0]
                boot_idx = np.where(~oob_mask)[0]

            X_boot = X_train.iloc[boot_idx][subset]
            y_boot = y_train[boot_idx]
            X_oob = X_train.iloc[oob_idx][subset]
            y_oob = y_train[oob_idx]

            best_model = None
            best_loss = np.inf
            best_params = None

            # Gridsearch on OOB
            for params in combos:
                clf = DecisionTreeClassifier(random_state=int(self.rng.integers(0, 2**31 - 1)), class_weight="balanced", **params)
                clf.fit(X_boot, y_boot)
                # Evaluate misclassification on OOB
                y_pred_oob = clf.predict(X_oob)
                err = 1.0 - float(accuracy_score(y_oob, y_pred_oob))
                if err < best_loss:
                    best_loss = err
                    best_model = clf
                    best_params = params

            # Compute prior logP for best model
            logP = self.prior_constructor.construct(best_model)
            self.trees.append(best_model)
            self.logPs.append(logP)

            # Validation loss (misclassification [0,1])
            y_pred_val = best_model.predict(X_val[best_model.feature_names_in_])
            loss_val = 1.0 - float(accuracy_score(y_val, y_pred_val))
            self.losses_val.append(loss_val)

        # Choose eta minimizing F(eta)
        logPs = np.array(self.logPs, dtype=float)
        losses = np.array(self.losses_val, dtype=float)
        best_eta = None
        best_F = np.inf
        for eta in self.eta_grid:
            eta = float(eta)
            # F = -(1/eta) * log(sum exp(logP - eta*loss))
            z = logPs - eta * losses
            m = np.max(z)
            lse = m + np.log(np.sum(np.exp(z - m)))
            F = -(1.0 / eta) * lse
            if F < best_F:
                best_F = F
                best_eta = eta
        self.best_eta = best_eta

        # Compute posterior weights q_i ∝ P(T) * exp(-eta*loss)
        z = logPs - best_eta * losses
        m = np.max(z)
        w = np.exp(z - m)
        self.q = w / w.sum()

    def predict(self, X_test):
        assert self.q is not None and len(self.trees) > 0, "Model not fitted."
        assert self.classes_ is not None
        K = len(self.classes_)
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)

        # Mixture of probabilities
        P = np.zeros((len(X_test_df), K), dtype=float)
        for qi, tree in zip(self.q, self.trees):
            # Align probabilities to global classes_
            proba = tree.predict_proba(X_test_df[tree.feature_names_in_])
            # tree.classes_ may be subset/order; align
            aligned = np.zeros((proba.shape[0], K), dtype=float)
            for j, cls in enumerate(tree.classes_):
                # find index in global classes_
                k = int(np.where(self.classes_ == cls)[0][0])
                aligned[:, k] = proba[:, j]
            P += qi * aligned

        y_hat = self.classes_[np.argmax(P, axis=1)]
        return y_hat