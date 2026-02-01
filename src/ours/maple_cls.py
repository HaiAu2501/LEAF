"""
Multi-Agent Prior Learning for Constructing Tree Ensembles (MAPLE) Classifier v5.

Major improvements over v4:
1. RF-style node-level feature subsampling (max_features on prior-filtered set)
2. Split validation: val_bandit + val_ensemble
3. Smoothed bandit rewards via EMA
4. Diversity feedback into prior (adaptive prior updates)
5. Error correlation for diversity instead of disagreement
6. Grid search over MAPLE-specific hyperparameters
"""
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from src.template import Algorithm
from src.ours.prior_factory import MultiAgentPriorFactory
from src.ours.bandit import UCB1, ThompsonSampling, EpsilonGreedy
from utils.logger import Logger


class MAPLEClassifier(Algorithm):
    """
    Multi-Agent Prior Learning for Constructing Tree Ensembles Classifier v5 with Grid Search.
    
    Supports grid search over both tree parameters and MAPLE-specific parameters:
    - feature_subset_ratio
    - diversity_weight  
    - val_bandit_ratio
    - ema_alpha
    - prior_update_rate
    """
    
    # MAPLE-specific parameters that can be searched
    MAPLE_PARAMS = {
        'feature_subset_ratio', 
        'diversity_weight', 
        'val_bandit_ratio', 
        'ema_alpha', 
        'prior_update_rate'
    }
    
    def __init__(
        self,
        logger: Logger,
        param_grid: dict[str, list] = None,
        n_agents: int = 3,
        exploration_coef: float = 0.5,
        bandit_type: str = "ucb",
        use_rf_style: bool = True,
        # Default values for searchable params (used if not in param_grid)
        feature_subset_ratio: float = 0.7,
        diversity_weight: float = 0.3,
        val_bandit_ratio: float = 0.3,
        ema_alpha: float = 0.2,
        prior_update_rate: float = 0.15,
    ):
        super().__init__(
            logger=logger,
            name="MAPLE_v5",
            task_type="classification",
            param_grid=param_grid,
        )
        # Fixed params
        self.n_agents = n_agents
        self.exploration_coef = exploration_coef
        self.bandit_type = bandit_type
        self.use_rf_style = use_rf_style
        
        # Default values for searchable params
        self._default_feature_subset_ratio = feature_subset_ratio
        self._default_diversity_weight = diversity_weight
        self._default_val_bandit_ratio = val_bandit_ratio
        self._default_ema_alpha = ema_alpha
        self._default_prior_update_rate = prior_update_rate
        
        # Will be set during fit
        self.prior_factory = None
        self.bandit = None
        self.trees = []
        self.tree_features = []
        self.tree_weights = []
        self.tree_agents = []
        self.classes_ = None
        self.rng = None
        
        # Best params found by grid search
        self.best_params_ = {}
    
    def setup(self, dataset, model):
        """Initialize multi-agent prior factory."""
        self.prior_factory = MultiAgentPriorFactory(
            dataset=dataset,
            model=model,
            n_agents=self.n_agents,
            logger=self.logger,
            prior_update_rate=self._default_prior_update_rate,
        )
        self.dataset = dataset
        self.model = model
    
    def _split_param_grid(self) -> tuple[list[dict], list[dict]]:
        """
        Split param_grid into MAPLE params and tree params.
        
        Returns:
            (maple_param_combos, tree_param_combos)
        """
        maple_grid = {}
        tree_grid = {}
        
        for key, values in self.param_grid.items():
            if key in self.MAPLE_PARAMS:
                maple_grid[key] = values
            elif key not in ('n_estimators', 'max_depth'):  # These come from config.yaml
                tree_grid[key] = values
        
        # Generate combinations
        def grid_to_combos(grid: dict) -> list[dict]:
            if not grid:
                return [{}]
            keys = list(grid.keys())
            vals = [grid[k] for k in keys]
            return [dict(zip(keys, prod)) for prod in itertools.product(*vals)]
        
        return grid_to_combos(maple_grid), grid_to_combos(tree_grid)
    
    def _get_maple_param(self, params: dict, name: str) -> float:
        """Get MAPLE param from dict or use default."""
        default_name = f'_default_{name}'
        return params.get(name, getattr(self, default_name))
    
    def fit(self, train: tuple, val: tuple, seed: int):
        """
        Fit MAPLE v5 with grid search over hyperparameters.
        
        Strategy:
        1. For each MAPLE param combination:
           - Train ensemble with those params
           - Evaluate on val_ensemble
        2. Select best MAPLE params
        3. Final model uses best params
        """
        X_train, y_train = train
        X_val_full, y_val_full = val
        
        assert isinstance(X_train, pd.DataFrame), "X_train must be DataFrame"
        assert isinstance(X_val_full, pd.DataFrame), "X_val must be DataFrame"
        
        self.classes_ = np.unique(y_train)
        
        # Split param grid
        maple_combos, tree_combos = self._split_param_grid()
        
        # If only one MAPLE combo, skip search
        if len(maple_combos) == 1:
            self._fit_single(
                train, val, seed,
                maple_params=maple_combos[0],
                tree_combos=tree_combos,
            )
            self.best_params_ = maple_combos[0]
            return
        
        # Grid search over MAPLE params
        best_score = -np.inf
        best_maple_params = maple_combos[0]
        
        print(f"Grid searching over {len(maple_combos)} MAPLE param combinations...")
        
        for i, maple_params in enumerate(tqdm(maple_combos)):
            # Reset prior factory for fresh start
            self.prior_factory.reset_all_priors()
            
            # Fit with this param combination
            score = self._fit_and_evaluate(
                train, val, seed + i,
                maple_params=maple_params,
                tree_combos=tree_combos,
            )
            
            if score > best_score:
                best_score = score
                best_maple_params = maple_params
                
            # if (i + 1) % 5 == 0:
            #     print(f"  {i+1}/{len(maple_combos)} done, best so far: {best_score:.4f}")
        
        print(f"Best MAPLE params: {best_maple_params} with score {best_score:.4f}")
        self.best_params_ = best_maple_params
        
        # Final fit with best params
        self.prior_factory.reset_all_priors()
        self._fit_single(
            train, val, seed,
            maple_params=best_maple_params,
            tree_combos=tree_combos,
        )
    
    def _fit_and_evaluate(
        self,
        train: tuple,
        val: tuple,
        seed: int,
        maple_params: dict,
        tree_combos: list[dict],
    ) -> float:
        """Fit with given params and return validation score."""
        self._fit_single(train, val, seed, maple_params, tree_combos)
        
        # Evaluate on full val set
        X_val, y_val = val
        y_pred = self.predict(X_val)
        return balanced_accuracy_score(y_val, y_pred)
    
    def _fit_single(
        self,
        train: tuple,
        val: tuple,
        seed: int,
        maple_params: dict,
        tree_combos: list[dict],
    ):
        """
        Fit a single MAPLE ensemble with given parameters.
        """
        self.trees = []
        self.tree_features = []
        self.tree_weights = []
        self.tree_agents = []
        self.rng = np.random.default_rng(seed)
        
        X_train, y_train = train
        X_val_full, y_val_full = val
        
        feature_names = list(X_train.columns)
        d = len(feature_names)
        N = len(X_train)
        
        # Get MAPLE params
        feature_subset_ratio = self._get_maple_param(maple_params, 'feature_subset_ratio')
        diversity_weight = self._get_maple_param(maple_params, 'diversity_weight')
        val_bandit_ratio = self._get_maple_param(maple_params, 'val_bandit_ratio')
        ema_alpha = self._get_maple_param(maple_params, 'ema_alpha')
        prior_update_rate = self._get_maple_param(maple_params, 'prior_update_rate')
        
        # Update prior factory's update rate
        for agent in self.prior_factory.agents:
            agent.prior_update_rate = prior_update_rate
        
        # Split validation set
        X_val_bandit, X_val_ens, y_val_bandit, y_val_ens = train_test_split(
            X_val_full, y_val_full,
            test_size=(1 - val_bandit_ratio),
            random_state=seed,
            stratify=y_val_full if len(np.unique(y_val_full)) > 1 else None
        )
        
        # Ensemble size from config (passed via alg.param_grid in config.yaml)
        n_estimators_list = self.param_grid.get("n_estimators", [100])
        n_trees = int(max(n_estimators_list))
        
        max_depth_list = self.param_grid.get("max_depth", [3])
        max_depth = int(max(max_depth_list))
        
        # Feature subset size
        m_prior = max(1, int(d * feature_subset_ratio))
        
        # RF-style max_features
        max_features_node = max(1, int(np.sqrt(m_prior))) if self.use_rf_style else None
        
        # Initialize bandit
        if self.bandit_type == "ts":
            self.bandit = ThompsonSampling(
                n_arms=self.n_agents, random_state=seed, ema_alpha=ema_alpha
            )
        elif self.bandit_type == "eps":
            self.bandit = EpsilonGreedy(
                n_arms=self.n_agents, random_state=seed, ema_alpha=ema_alpha
            )
        else:
            self.bandit = UCB1(
                n_arms=self.n_agents,
                exploration_coef=self.exploration_coef,
                random_state=seed,
                ema_alpha=ema_alpha,
                use_risk_adjusted=True,
                risk_coef=0.3,
            )
        
        # Train trees
        val_ens_accs = []
        val_ens_errors = []
        good_tree_threshold = 0.55
        
        for i in range(n_trees):
            # 1. Bandit selects agent
            agent_id = self.bandit.select_arm()
            agent = self.prior_factory.get_agent(agent_id)
            
            # 2. Sample features
            feat_indices = agent.sample_features(m_prior, self.rng)
            feat_names_subset = [feature_names[j] for j in feat_indices]
            
            # 3. Bootstrap sample
            boot_idx = self.rng.integers(0, N, size=N)
            X_boot = X_train.iloc[boot_idx][feat_names_subset]
            y_boot = y_train[boot_idx]
            
            # 4. Select tree params (cycle through or pick best from previous)
            tree_params = tree_combos[i % len(tree_combos)]
            
            # 5. Train tree
            tree = DecisionTreeClassifier(
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                class_weight="balanced",
                max_features=max_features_node,
                max_depth=max_depth,
                **tree_params
            )
            tree.fit(X_boot, y_boot)
            
            # 6. Evaluate
            y_pred_bandit = tree.predict(X_val_bandit[feat_names_subset])
            val_bandit_acc = balanced_accuracy_score(y_val_bandit, y_pred_bandit)
            
            y_pred_ens = tree.predict(X_val_ens[feat_names_subset])
            val_ens_acc = balanced_accuracy_score(y_val_ens, y_pred_ens)
            error_vector = (y_pred_ens != y_val_ens).astype(float)
            
            # 7. Store
            self.trees.append(tree)
            self.tree_features.append(feat_names_subset)
            self.tree_agents.append(agent_id)
            val_ens_accs.append(val_ens_acc)
            val_ens_errors.append(error_vector)
            
            # 8. Update bandit
            reward = np.clip((val_bandit_acc - 0.5) * 2, 0.0, 1.0)
            self.bandit.update(agent_id, reward)
            
            # 9. Record for prior update
            used_in_tree = np.where(tree.feature_importances_ > 0)[0].tolist()
            is_good = val_bandit_acc > good_tree_threshold
            self.prior_factory.record_tree_result(
                agent_id=agent_id,
                reward=val_bandit_acc,
                used_features=[feat_indices[j] for j in used_in_tree if j < len(feat_indices)],
                is_good_tree=is_good,
            )
        
        # Compute error-correlation weights
        val_ens_accs = np.array(val_ens_accs)
        val_ens_errors = np.array(val_ens_errors)
        
        n_trees_actual = len(self.trees)
        error_corr = np.zeros((n_trees_actual, n_trees_actual))
        
        for i in range(n_trees_actual):
            for j in range(i + 1, n_trees_actual):
                corr = np.corrcoef(val_ens_errors[i], val_ens_errors[j])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                error_corr[i, j] = corr
                error_corr[j, i] = corr
        
        mean_corr = error_corr.sum(axis=1) / (n_trees_actual - 1 + 1e-8)
        diversity_scores = 1 - mean_corr
        
        # Normalize and combine
        acc_range = val_ens_accs.max() - val_ens_accs.min()
        div_range = diversity_scores.max() - diversity_scores.min()
        
        acc_norm = (val_ens_accs - val_ens_accs.min()) / (acc_range + 1e-8)
        div_norm = (diversity_scores - diversity_scores.min()) / (div_range + 1e-8)
        
        combined = (1 - diversity_weight) * acc_norm + diversity_weight * div_norm
        
        # Softmax weights
        temperature = 0.5
        weights = np.exp(combined / temperature)
        weights = weights / weights.sum()
        
        self.tree_weights = weights.tolist()
    
    def predict(self, X_test) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X_test) -> np.ndarray:
        """Predict class probabilities."""
        if len(self.trees) == 0:
            raise ValueError("Model not fitted.")
        
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        K = len(self.classes_)
        n_samples = len(X_test_df)
        
        P = np.zeros((n_samples, K), dtype=float)
        
        for tree, feats, weight in zip(self.trees, self.tree_features, self.tree_weights):
            proba = tree.predict_proba(X_test_df[feats])
            
            aligned = np.zeros((n_samples, K), dtype=float)
            for j, cls in enumerate(tree.classes_):
                k = int(np.where(self.classes_ == cls)[0][0])
                aligned[:, k] = proba[:, j]
            
            P += weight * aligned
        
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        return P