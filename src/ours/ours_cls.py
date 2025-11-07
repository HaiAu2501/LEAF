import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.template import Algorithm
from src.ours.prior_constructor import PriorConstructor
from utils.logger import Logger


class OursClassifier(Algorithm):
    def __init__(
        self,
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
        self.trees = []
        self.logPs = []
        self.losses_val = []
        self.q = None
        self.classes_ = None
        self.rng = None

    def _bounded_log_loss(self, y_true, proba, classes):
        """Normalized cross-entropy loss in [0,1].
        loss = (1/ log(K)) * E[-log p(y|x)] where K is number of global classes.
        Args:
            y_true: array-like of shape (N,)
            proba: array-like (N, K_tree) predicted probabilities from tree
            classes: array-like (K_tree,) classes ordering for columns in proba
        Returns:
            float in [0,1]
        """
        y_true = np.asarray(y_true)
        proba = np.asarray(proba, dtype=float)
        eps = 1e-12
        # Map class -> column index for this tree
        col_map = {c: i for i, c in enumerate(classes)}
        # Extract probability assigned to true class for each sample
        p_true = np.array([proba[i, col_map[y]] if y in col_map else eps for i, y in enumerate(y_true)], dtype=float)
        p_true = np.clip(p_true, eps, 1.0)
        K = len(self.classes_) if self.classes_ is not None else len(classes)
        logK = np.log(max(K, 2))  # prevent div by zero, K>=2 for classification
        loss = -np.mean(np.log(p_true)) / logK
        # Numerical guard: clip into [0,1]
        return float(np.clip(loss, 0.0, 1.0))
    
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
        self.tree_feats = []
        self.rng = np.random.default_rng(seed)

        X_train, y_train = train
        X_val, y_val = val
        assert isinstance(X_train, pd.DataFrame) and isinstance(X_val, pd.DataFrame), "Expect DataFrame inputs"
        # Set global classes for alignment in predict
        self.classes_ = np.unique(y_train)

        # Determine number of trees from param_grid (use max n_estimators if provided)
        n_estimators_list = self.param_grid.get("n_estimators", [50]) if self.param_grid else [50]
        n_trees = int(max(n_estimators_list)) if len(n_estimators_list) > 0 else 50

        # Build candidate hyperparam combos excluding n_estimators
        grid_no_n = {k: v for k, v in (self.param_grid or {}).items() if k != "n_estimators"}
        if len(grid_no_n) == 0:
            combos = [dict()]
        else:
            keys = list(grid_no_n.keys())
            vals = [grid_no_n[k] for k in keys]
            combos = [dict(zip(keys, prod)) for prod in itertools.product(*vals)]

        # Feature prior probabilities over all features
        fw = self.prior_constructor.feature_weights
        feat_names = list(X_train.columns)
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

            # 2) Bootstrap rows for training
            boot_idx = self.rng.integers(0, N, size=N)
            X_boot = X_train.iloc[boot_idx][subset]
            y_boot = y_train[boot_idx]

            best_model = None
            best_loss = np.inf

            # Gridsearch on validation set using the sampled subset
            for params in combos:
                clf = DecisionTreeClassifier(random_state=int(self.rng.integers(0, 2**31 - 1)), class_weight="balanced", **params)
                clf.fit(X_boot, y_boot)
                proba_val_gs = clf.predict_proba(X_val[subset])
                err = self._bounded_log_loss(y_val, proba_val_gs, clf.classes_)
                if err < best_loss:
                    best_loss = err
                    best_model = clf

            # Save tree and its feature list
            self.trees.append(best_model)
            self.tree_feats.append(subset)

            # Compute prior logP for the selected tree
            logP = self.prior_constructor.construct(best_model, feature_names=subset)
            self.logPs.append(logP)

            # Final validation loss with best tree
            proba_val = best_model.predict_proba(X_val[subset])
            loss_val = self._bounded_log_loss(y_val, proba_val, best_model.classes_)
            self.losses_val.append(loss_val)

        # Compute posterior weights q_i ∝ P(T) * exp(-eta*loss) with eta=1
        logPs = np.array(self.logPs, dtype=float)
        losses = np.array(self.losses_val, dtype=float)
        z = logPs - losses
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
        for qi, tree, feats in zip(self.q, self.trees, self.tree_feats):
            # Align probabilities to global classes_
            proba = tree.predict_proba(X_test_df[feats])
            # tree.classes_ may be subset/order; align
            aligned = np.zeros((proba.shape[0], K), dtype=float)
            for j, cls in enumerate(tree.classes_):
                # find index in global classes_
                k = int(np.where(self.classes_ == cls)[0][0])
                aligned[:, k] = proba[:, j]
            P += qi * aligned

        y_hat = self.classes_[np.argmax(P, axis=1)]
        return y_hat