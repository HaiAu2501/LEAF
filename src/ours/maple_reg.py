"""
Multi-Agent Prior Learning for Constructing Tree Ensembles (MAPLE) Regressor v6.

Regression adaptation with key differences from classifier:
1. Reward = OOB clipped-R² (bounded to [0, 1])
2. Quality for tree weights = OOB clipped-R² (same scale as bandit)
3. Diversity = 1 - |corr(predictions)| on common OOB samples
4. Bootstrap = regular (not stratified)
5. Hardness = |residual| / std(y) (normalized residual magnitude)
6. No calibration (not applicable to regression)
"""
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.template import Algorithm
from src.ours.prior_factory import MultiAgentPriorFactory
from src.ours.bandit import UCB1, ThompsonSampling, EpsilonGreedy
from utils.logger import Logger


class MAPLERegressor(Algorithm):
    """
    Multi-Agent Prior Learning for Constructing Tree Ensembles Regressor v6.
    
    Key adaptations for regression:
    - R² based rewards instead of log-loss
    - Correlation-based diversity instead of disagreement
    - Regular bootstrap instead of stratified
    - Residual-based hardness metric
    """
    
    MAPLE_PARAMS = {
        'feature_subset_ratio', 
        'diversity_weight', 
        'ema_alpha', 
        'prior_update_rate',
        'prior_temperature',
        'uniform_mix_ratio',
        'weight_uniform_mix',
    }
    
    def __init__(
        self,
        logger: Logger,
        param_grid: dict[str, list] = None,
        n_agents: int = 3,
        exploration_coef: float = 0.5,
        bandit_type: str = "ucb",
        use_rf_style: bool = True,
        # Default values for searchable params
        feature_subset_ratio: float = 0.7,
        diversity_weight: float = 0.2,
        ema_alpha: float = 0.2,
        prior_update_rate: float = 0.15,
        prior_temperature: float = 1.5,
        uniform_mix_ratio: float = 0.2,
        weight_uniform_mix: float = 0.1,
    ):
        super().__init__(
            logger=logger,
            name="MAPLE_v6",
            task_type="regression",
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
        self._default_ema_alpha = ema_alpha
        self._default_prior_update_rate = prior_update_rate
        self._default_prior_temperature = prior_temperature
        self._default_uniform_mix_ratio = uniform_mix_ratio
        self._default_weight_uniform_mix = weight_uniform_mix
        
        # Will be set during fit
        self.prior_factory = None
        self.bandit = None
        self.trees = []
        self.tree_features = []
        self.tree_weights = []
        self.tree_agents = []
        self.rng = None
        
        # OOB tracking
        self.oob_predictions = None  # Shape: (n_samples, n_trees)
        self.oob_counts = None       # How many trees have OOB prediction for each sample
        
        # Target statistics for normalization
        self.y_mean_ = None
        self.y_std_ = None
        
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
        """Split param_grid into MAPLE params and tree params."""
        maple_grid = {}
        tree_grid = {}
        
        for key, values in self.param_grid.items():
            if key in self.MAPLE_PARAMS:
                maple_grid[key] = values
            elif key not in ('n_estimators', 'max_depth'):
                tree_grid[key] = values
        
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
    
    def _bootstrap(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create regular bootstrap sample.
        
        Returns:
            (bootstrap_indices, oob_indices)
        """
        n = len(X)
        
        # Sample with replacement
        boot_indices = rng.choice(n, size=n, replace=True)
        
        # OOB: samples not in bootstrap
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_indices)] = False
        oob_indices = np.where(oob_mask)[0]
        
        return boot_indices, oob_indices
    
    def _compute_normalized_r2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        global_var: float,
    ) -> float:
        """
        Compute R²-like score normalized by GLOBAL variance.
        
        reward = 1 - MSE / var_global
        
        This is more stable than standard R² because:
        1. Standard R² uses local variance of y_true, which varies per OOB subset
        2. Trees with low-variance OOB subsets get extremely noisy R² scores
        3. Using global variance makes rewards comparable across trees
        
        Args:
            y_true: True target values (OOB subset)
            y_pred: Predicted values
            global_var: Variance of full training set (constant across trees)
            
        Returns:
            Normalized R² in [0, 1]
        """
        if len(y_true) < 2:
            return 0.0
        
        mse = np.mean((y_true - y_pred) ** 2)
        
        # R²-like but with global variance (stable denominator)
        r2_normalized = 1.0 - mse / global_var
        
        return float(np.clip(r2_normalized, 0.0, 1.0))
    
    def _compute_hardness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hardness for each sample: |residual| / std(y).
        
        Higher hardness = model was more wrong (larger residual).
        Normalized by target std for scale invariance.
        """
        residuals = np.abs(y_true - y_pred)
        
        # Normalize by target std (use stored value or compute)
        if self.y_std_ is not None and self.y_std_ > 1e-8:
            hardness = residuals / self.y_std_
        else:
            std = np.std(y_true)
            hardness = residuals / (std + 1e-8)
        
        # Clip to reasonable range [0, 3] (3 std deviations)
        hardness = np.clip(hardness, 0.0, 3.0) / 3.0  # Normalize to [0, 1]
        
        return hardness
    
    def _sample_features_with_tempering(
        self,
        agent_id: int,
        subset_size: int,
        rng: np.random.Generator,
        temperature: float,
        uniform_mix: float,
    ) -> list[int]:
        """
        Sample features from prior with tempering and uniform mixing.
        
        p_f ∝ (prior_f + ε)^(1/τ), then mix with uniform.
        """
        agent = self.prior_factory.get_agent(agent_id)
        d = agent.d
        
        # Get prior probabilities and apply tempering
        prior_probs = agent.prior_probs
        tempered = np.power(prior_probs + 1e-6, 1.0 / max(temperature, 0.1))
        tempered = tempered / tempered.sum()
        
        # Mix with uniform
        uniform = np.ones(d) / d
        final_probs = (1 - uniform_mix) * tempered + uniform_mix * uniform
        final_probs = final_probs / final_probs.sum()
        
        # Sample
        subset_size = min(subset_size, d)
        indices = rng.choice(d, size=subset_size, replace=False, p=final_probs)
        
        return indices.tolist()
    
    def _compute_residual_diversity(
        self,
        predictions: list[np.ndarray],
        y_true: np.ndarray,
        buffer_size: int = 20,
    ) -> np.ndarray:
        """
        Compute residual-based diversity scores (error diversity).
        
        e_i = y - pred_i (residuals)
        div(i) = 1 - mean(|corr(e_i, e_j)|) for j != i
        
        This encourages trees that make DIFFERENT errors, which is the true
        goal of ensemble diversity. Trees with uncorrelated errors will
        cancel out when averaged.
        
        Note: Measuring diversity on predictions would penalize good trees
        that all track the true signal (high pred correlation ≠ bad).
        
        Returns diversity score for each tree (lower error correlation = higher diversity).
        """
        n_trees = len(predictions)
        if n_trees <= 1:
            return np.ones(n_trees)
        
        # Use only last buffer_size trees for efficiency
        start_idx = max(0, n_trees - buffer_size)
        
        diversity_scores = np.zeros(n_trees)
        
        for i in range(n_trees):
            error_correlations = []
            
            # Compare with trees in buffer
            for j in range(start_idx, n_trees):
                if i != j:
                    # Find common OOB samples (both have predictions)
                    mask = ~(np.isnan(predictions[i]) | np.isnan(predictions[j]))
                    if mask.sum() > 2:  # Need at least 3 points for meaningful correlation
                        # Compute residuals
                        e_i = y_true[mask] - predictions[i][mask]
                        e_j = y_true[mask] - predictions[j][mask]
                        
                        # Compute Pearson correlation of residuals
                        if np.std(e_i) > 1e-8 and np.std(e_j) > 1e-8:
                            corr = np.corrcoef(e_i, e_j)[0, 1]
                            error_correlations.append(np.abs(corr))
            
            if error_correlations:
                # Diversity = 1 - mean absolute error correlation
                # High diversity = trees make different mistakes
                diversity_scores[i] = 1.0 - np.mean(error_correlations)
            else:
                diversity_scores[i] = 0.5  # Neutral diversity
        
        return diversity_scores
    
    def fit(self, train: tuple, val: tuple, seed: int):
        """
        Fit MAPLE v6 Regressor with OOB-based evaluation.
        
        Note: val is still passed for API compatibility but not used for tree evaluation.
        Val is only used for final hyperparameter selection if grid searching.
        """
        X_train, y_train = train
        X_val, y_val = val
        
        # Flatten y if needed
        y_train = np.asarray(y_train).flatten()
        y_val = np.asarray(y_val).flatten()
        
        # Combine train and val for full training (pure bagging uses all data)
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = np.concatenate([y_train, y_val])
        
        assert isinstance(X_full, pd.DataFrame), "X must be DataFrame"
        
        # Store target statistics for normalization
        self.y_mean_ = np.mean(y_full)
        self.y_std_ = np.std(y_full)
        
        # Split param grid
        maple_combos, tree_combos = self._split_param_grid()
        
        # If only one MAPLE combo, skip search
        if len(maple_combos) == 1:
            self._fit_single(
                X_full, y_full, seed,
                maple_params=maple_combos[0],
                tree_combos=tree_combos,
            )
            self.best_params_ = maple_combos[0]
            return
        
        # Grid search using OOB performance
        best_score = -np.inf
        best_maple_params = maple_combos[0]
        
        print(f"Grid searching over {len(maple_combos)} MAPLE param combinations...")
        
        for i, maple_params in enumerate(tqdm(maple_combos)):
            self.prior_factory.reset_all_priors()
            
            # Fit and get OOB score
            self._fit_single(
                X_full, y_full, seed,
                maple_params=maple_params,
                tree_combos=tree_combos,
            )
            
            # Evaluate using aggregated OOB predictions
            score = self._get_oob_score(y_full)
            
            if score > best_score:
                best_score = score
                best_maple_params = maple_params
        
        print(f"Best MAPLE params: {best_maple_params} with OOB R² {best_score:.4f}")
        self.best_params_ = best_maple_params
        
        # Final fit with best params
        self.prior_factory.reset_all_priors()
        self._fit_single(
            X_full, y_full, seed,
            maple_params=best_maple_params,
            tree_combos=tree_combos,
        )
    
    def _get_oob_score(self, y_true: np.ndarray) -> float:
        """Compute R² from aggregated OOB predictions."""
        n_samples = len(y_true)
        
        # Aggregate OOB predictions (weighted by tree weights)
        oob_pred = np.zeros(n_samples)
        oob_weight_sum = np.zeros(n_samples)
        
        for t, weight in enumerate(self.tree_weights):
            for i in range(n_samples):
                if self.oob_counts[i, t] > 0:
                    oob_pred[i] += weight * self.oob_predictions[i, t]
                    oob_weight_sum[i] += weight
        
        # Samples with OOB predictions
        valid_mask = oob_weight_sum > 0
        if valid_mask.sum() < 2:
            return 0.0
        
        oob_pred[valid_mask] /= oob_weight_sum[valid_mask]
        
        return r2_score(y_true[valid_mask], oob_pred[valid_mask])
    
    def _fit_single(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        seed: int,
        maple_params: dict,
        tree_combos: list[dict],
    ):
        """Fit a single MAPLE ensemble with OOB evaluation."""
        self.trees = []
        self.tree_features = []
        self.tree_weights = []
        self.tree_agents = []
        self.rng = np.random.default_rng(seed)
        
        feature_names = list(X_train.columns)
        d = len(feature_names)
        N = len(X_train)
        
        # Global variance for normalized R² (stable across OOB subsets)
        # Since y is standardized, this should be ~1, but compute for safety
        global_var = np.var(y_train) + 1e-12
        
        # Get MAPLE params
        feature_subset_ratio = self._get_maple_param(maple_params, 'feature_subset_ratio')
        diversity_weight = self._get_maple_param(maple_params, 'diversity_weight')
        ema_alpha = self._get_maple_param(maple_params, 'ema_alpha')
        prior_update_rate = self._get_maple_param(maple_params, 'prior_update_rate')
        prior_temperature = self._get_maple_param(maple_params, 'prior_temperature')
        uniform_mix_ratio = self._get_maple_param(maple_params, 'uniform_mix_ratio')
        weight_uniform_mix = self._get_maple_param(maple_params, 'weight_uniform_mix')
        
        # Update prior factory's update rate
        for agent in self.prior_factory.agents:
            agent.prior_update_rate = prior_update_rate
        
        # Ensemble size
        n_estimators_list = self.param_grid.get("n_estimators", [100])
        n_trees = int(max(n_estimators_list))
        
        max_depth_list = self.param_grid.get("max_depth", [3])
        max_depth = int(max(max_depth_list))
        
        # Feature subset size
        m_prior = max(1, int(d * feature_subset_ratio))
        
        # RF-style max_features at node level
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
        
        # OOB tracking
        self.oob_predictions = np.zeros((N, n_trees), dtype=float)
        self.oob_counts = np.zeros((N, n_trees), dtype=int)
        
        # For diversity calculation
        oob_predictions_list = []  # List of arrays, each tree's predictions (NaN for non-OOB)
        oob_r2_scores = []
        
        # Temperature schedule: start high (more exploration), decay
        temp_schedule = lambda t: max(1.0, prior_temperature * (0.95 ** (t / 20)))
        
        for t in range(n_trees):
            # 1. Bandit selects agent
            agent_id = self.bandit.select_arm()
            
            # 2. Sample features with tempering
            current_temp = temp_schedule(t)
            feat_indices = self._sample_features_with_tempering(
                agent_id, m_prior, self.rng, current_temp, uniform_mix_ratio
            )
            feat_names_subset = [feature_names[j] for j in feat_indices]
            
            # 3. Regular bootstrap (not stratified)
            boot_indices, oob_indices = self._bootstrap(X_train, y_train, self.rng)
            
            X_boot = X_train.iloc[boot_indices][feat_names_subset]
            y_boot = y_train[boot_indices]
            
            # 4. Select tree params
            tree_params = tree_combos[t % len(tree_combos)]
            
            # 5. Train tree
            tree = DecisionTreeRegressor(
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                max_features=max_features_node,
                max_depth=max_depth,
                **tree_params
            )
            tree.fit(X_boot, y_boot)
            
            # 6. OOB evaluation
            if len(oob_indices) > 0:
                X_oob = X_train.iloc[oob_indices][feat_names_subset]
                y_oob = y_train[oob_indices]
                
                # Get predictions
                oob_pred = tree.predict(X_oob)
                
                # Store OOB predictions
                for idx_local, idx_global in enumerate(oob_indices):
                    self.oob_predictions[idx_global, t] = oob_pred[idx_local]
                    self.oob_counts[idx_global, t] = 1
                
                # Compute normalized R² as reward (uses global variance for stability)
                reward = self._compute_normalized_r2(y_oob, oob_pred, global_var)
                
                # Compute hardness (normalized residuals)
                hardness = self._compute_hardness(y_oob, oob_pred)
                
                # Store normalized R² for tree weighting (same metric as bandit)
                oob_r2_scores.append(reward)
                
                # Predictions for diversity (NaN for non-OOB samples)
                oob_pred_full = np.full(N, np.nan)
                oob_pred_full[oob_indices] = oob_pred
                oob_predictions_list.append(oob_pred_full)
            else:
                # No OOB samples (rare)
                reward = 0.0
                hardness = np.array([])
                oob_r2_scores.append(0.0)
                oob_predictions_list.append(np.full(N, np.nan))
            
            # 7. Store tree
            self.trees.append(tree)
            self.tree_features.append(feat_names_subset)
            self.tree_agents.append(agent_id)
            
            # 8. Update bandit with hardness-scaled reward
            if len(hardness) > 0:
                mean_hardness = float(np.clip(np.mean(hardness), 0.0, 1.0))
                # Scale reward: good performance on hard samples is more valuable
                reward_eff = reward * (1.0 + mean_hardness)
                reward_eff = float(np.clip(reward_eff, 0.0, 1.0))
            else:
                mean_hardness = 0.5
                reward_eff = reward
            
            self.bandit.update(agent_id, reward_eff)
            
            # 9. Update prior with hardness-weighted feature usage
            if len(oob_indices) > 0:
                used_features_mask = tree.feature_importances_ > 0
                used_in_tree = np.where(used_features_mask)[0].tolist()
                
                # Map back to global feature indices
                global_used = [feat_indices[j] for j in used_in_tree if j < len(feat_indices)]
                
                # Pass hardness-scaled reward to prior factory
                self.prior_factory.record_tree_result(
                    agent_id=agent_id,
                    reward=reward_eff,
                    used_features=global_used,
                    is_good_tree=(reward_eff > 0.3),
                    hardness=mean_hardness,
                )
        
        # Compute tree weights based on OOB normalized-R² and error diversity
        # Note: Using same metric (normalized R²) for both bandit and quality ensures consistency
        oob_r2_scores = np.array(oob_r2_scores)
        
        # Diversity scores (residual-based - measures error diversity)
        diversity_scores = self._compute_residual_diversity(oob_predictions_list, y_train)
        
        # Normalize scores
        r2_range = oob_r2_scores.max() - oob_r2_scores.min()
        div_range = diversity_scores.max() - diversity_scores.min()
        
        # Higher normalized R² = better (already in [0, 1] with stable scale)
        if r2_range > 1e-8:
            quality_score = (oob_r2_scores - oob_r2_scores.min()) / r2_range
        else:
            quality_score = np.ones(n_trees)
        
        if div_range > 1e-8:
            div_score = (diversity_scores - diversity_scores.min()) / div_range
        else:
            div_score = np.ones(n_trees)
        
        # Combined score
        combined = (1 - diversity_weight) * quality_score + diversity_weight * div_score
        
        # Softmax weights with temperature
        temperature = 0.5
        weights = np.exp(combined / temperature)
        weights = weights / weights.sum()
        
        # Mix with uniform to avoid collapse
        uniform_weights = np.ones(n_trees) / n_trees
        weights = (1 - weight_uniform_mix) * weights + weight_uniform_mix * uniform_weights
        
        self.tree_weights = weights.tolist()
    
    def predict(self, X_test) -> np.ndarray:
        """Predict target values."""
        if len(self.trees) == 0:
            raise ValueError("Model not fitted.")
        
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        n_samples = len(X_test_df)
        
        predictions = np.zeros(n_samples, dtype=float)
        
        for tree, feats, weight in zip(self.trees, self.tree_features, self.tree_weights):
            pred = tree.predict(X_test_df[feats])
            predictions += weight * pred
        
        return predictions