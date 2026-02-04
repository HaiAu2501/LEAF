"""
Multi-Agent Prior Learning for Constructing Tree Ensembles (MAPLE) Regressor v5.

Regression variant adapted from MAPLEClassifier.

Key differences from classification:
1. Uses DecisionTreeRegressor instead of DecisionTreeClassifier
2. Uses RMSE/R² for evaluation instead of balanced accuracy
3. Weighted average prediction instead of probability voting
4. Bandit rewards based on negative RMSE (normalized)
5. No class balancing needed
"""
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.template import Algorithm
from src.ours.prior_factory import MultiAgentPriorFactory
from src.ours.bandit import UCB1, ThompsonSampling, EpsilonGreedy
from utils.logger import Logger


class MAPLERegressor(Algorithm):
    """
    Multi-Agent Prior Learning for Constructing Tree Ensembles Regressor v5 with Grid Search.
    
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
        # Default values for searchable params (tuned for regression)
        feature_subset_ratio: float = 0.7,
        diversity_weight: float = 0.15,  # Lower than classification
        val_bandit_ratio: float = 0.2,   # Smaller for regression datasets
        ema_alpha: float = 0.2,
        prior_update_rate: float = 0.1,
    ):
        super().__init__(
            logger=logger,
            name="MAPLE_v5",
            task_type="regression",
            param_grid=param_grid,
        )
        # Fixed params
        self.n_agents = n_agents
        self.exploration_coef = exploration_coef
        self.bandit_type = bandit_type
        self.use_rf_style = use_rf_style
        
        # Default values for searchable params
        # Note: Regression datasets are often smaller, so use smaller val_bandit_ratio
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
        self.rng = None
        
        # For reward normalization
        self._y_std = 1.0
        
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
        Fit MAPLE v5 Regressor with grid search over hyperparameters.
        
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
        
        # Store y_std for reward normalization
        self._y_std = max(np.std(y_train), 1e-8)
        
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
        best_score = np.inf  # Lower RMSE is better
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
            
            if score < best_score:  # Lower RMSE is better
                best_score = score
                best_maple_params = maple_params
        
        print(f"Best MAPLE params: {best_maple_params} with RMSE {best_score:.4f}")
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
        """Fit with given params and return validation RMSE (lower is better)."""
        self._fit_single(train, val, seed, maple_params, tree_combos)
        
        # Evaluate on full val set
        X_val, y_val = val
        y_pred = self.predict(X_val)
        return root_mean_squared_error(y_val, y_pred)
    
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
        
        # Flatten y if needed
        y_train = np.asarray(y_train).ravel()
        y_val_full = np.asarray(y_val_full).ravel()
        
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
        )
        y_val_bandit = np.asarray(y_val_bandit).ravel()
        y_val_ens = np.asarray(y_val_ens).ravel()
        
        # Ensemble size from config (passed via alg.param_grid in config.yaml)
        n_estimators_list = self.param_grid.get("n_estimators", [100])
        n_trees = int(max(n_estimators_list))
        
        max_depth_list = self.param_grid.get("max_depth", [3])
        max_depth = int(max(max_depth_list))
        
        # Feature subset size
        m_prior = max(1, int(d * feature_subset_ratio))
        
        # RF-style max_features (use d/3 for regression, not sqrt like classification)
        if self.use_rf_style:
            max_features_node = max(1, int(m_prior / 3))
        else:
            max_features_node = None
        
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
        val_ens_r2s = []
        val_ens_preds = []  # Store predictions for diversity calculation
        
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
            tree = DecisionTreeRegressor(
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                max_features=max_features_node,
                max_depth=max_depth,
                **tree_params
            )
            tree.fit(X_boot, y_boot)
            
            # 6. Evaluate on bandit validation set using R²
            y_pred_bandit = tree.predict(X_val_bandit[feat_names_subset])
            val_bandit_r2 = r2_score(y_val_bandit, y_pred_bandit)
            
            # Evaluate on ensemble validation set
            y_pred_ens = tree.predict(X_val_ens[feat_names_subset])
            val_ens_r2 = r2_score(y_val_ens, y_pred_ens)
            
            # 7. Store
            self.trees.append(tree)
            self.tree_features.append(feat_names_subset)
            self.tree_agents.append(agent_id)
            val_ens_r2s.append(val_ens_r2)
            val_ens_preds.append(y_pred_ens)
            
            # 8. Update bandit with R²-based reward
            # R² is naturally in (-∞, 1], we clip to [0, 1] for bandit
            # R² = 0 means no better than mean, R² = 1 is perfect
            # Negative R² means worse than mean prediction
            reward = np.clip((val_bandit_r2 + 1) / 2, 0.0, 1.0)  # Map [-1, 1] -> [0, 1]
            self.bandit.update(agent_id, reward)
            
            # 9. Record for prior update
            used_in_tree = np.where(tree.feature_importances_ > 0)[0].tolist()
            # Consider "good tree" if R² > 0.1 (explains at least 10% variance)
            is_good = val_bandit_r2 > 0.1
            self.prior_factory.record_tree_result(
                agent_id=agent_id,
                reward=reward,
                used_features=[feat_indices[j] for j in used_in_tree if j < len(feat_indices)],
                is_good_tree=is_good,
            )
        
        # Compute diversity based on prediction disagreement (complementarity)
        val_ens_r2s = np.array(val_ens_r2s)
        val_ens_preds = np.array(val_ens_preds)  # Shape: (n_trees, n_val_ens)
        
        n_trees_actual = len(self.trees)
        
        # Diversity: measure how different each tree's predictions are from ensemble mean
        # Trees that make different errors are more valuable
        ensemble_mean_pred = val_ens_preds.mean(axis=0)
        
        # For each tree, compute how much it disagrees with ensemble (excluding itself)
        diversity_scores = np.zeros(n_trees_actual)
        for i in range(n_trees_actual):
            # Leave-one-out ensemble mean
            other_preds = np.delete(val_ens_preds, i, axis=0)
            if len(other_preds) > 0:
                loo_mean = other_preds.mean(axis=0)
                # Disagreement = variance of this tree's prediction from LOO ensemble
                diversity_scores[i] = np.mean((val_ens_preds[i] - loo_mean) ** 2)
            else:
                diversity_scores[i] = 0.0
        
        # Normalize diversity scores
        div_range = diversity_scores.max() - diversity_scores.min() + 1e-8
        div_norm = (diversity_scores - diversity_scores.min()) / div_range
        
        # Normalize R² scores (higher is better)
        # Clip very negative R² to avoid outliers dominating
        r2_clipped = np.clip(val_ens_r2s, -1.0, 1.0)
        r2_min = r2_clipped.min()
        r2_max = r2_clipped.max()
        r2_range = r2_max - r2_min + 1e-8
        r2_norm = (r2_clipped - r2_min) / r2_range
        
        # Combine accuracy (R²) and diversity
        combined = (1 - diversity_weight) * r2_norm + diversity_weight * div_norm
        
        # Softmax weights
        temperature = 0.5
        weights = np.exp(combined / temperature)
        weights = weights / weights.sum()
        
        self.tree_weights = weights.tolist()
    
    def predict(self, X_test) -> np.ndarray:
        """Predict target values using weighted average of trees."""
        if len(self.trees) == 0:
            raise ValueError("Model not fitted.")
        
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        n_samples = len(X_test_df)
        
        predictions = np.zeros(n_samples, dtype=float)
        
        for tree, feats, weight in zip(self.trees, self.tree_features, self.tree_weights):
            pred = tree.predict(X_test_df[feats])
            predictions += weight * pred
        
        return predictions