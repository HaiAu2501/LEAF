"""
Multi-Agent Prior Learning for Constructing Tree Ensembles (MAPLE) Classifier v6.

Major improvements over v5 (OOB-based pure bagging):
1. Use OOB (Out-of-Bag) evaluation instead of train/val split - train on full data
2. Bandit reward: log-loss improvement on OOB (continuous, better signal for binary)
3. Stratified bootstrap for handling class imbalance
4. Prior update based on hardness-weighted feature usage (not fixed threshold)
5. Feature sampling with tempering + uniform mixing (avoid prior collapse)
6. Diversity: disagreement rate instead of Pearson correlation (more stable)
7. Tree weighting: OOB logloss + diversity with uniform mixing (avoid softmax collapse)
8. Optional OOB calibration via Platt scaling
"""
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.template import Algorithm
from src.ours.prior_factory import MultiAgentPriorFactory
from src.ours.bandit import UCB1, ThompsonSampling, EpsilonGreedy
from utils.logger import Logger


class MAPLEClassifier(Algorithm):
    """
    Multi-Agent Prior Learning for Constructing Tree Ensembles Classifier v6.
    
    Key changes from v5:
    - Pure bagging with OOB evaluation (no train/val split)
    - Log-loss based rewards for better bandit signal
    - Hardness-weighted prior updates
    - Disagreement-based diversity
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
        use_calibration: bool = False,
        # Default values for searchable params
        feature_subset_ratio: float = 0.7,
        diversity_weight: float = 0.2,
        ema_alpha: float = 0.2,
        prior_update_rate: float = 0.15,
        prior_temperature: float = 1.5,      # Tempering for prior sampling
        uniform_mix_ratio: float = 0.2,      # Mix with uniform for exploration
        weight_uniform_mix: float = 0.1,     # Mix tree weights with uniform
    ):
        super().__init__(
            logger=logger,
            name="MAPLE_v6",
            task_type="classification",
            param_grid=param_grid,
        )
        # Fixed params
        self.n_agents = n_agents
        self.exploration_coef = exploration_coef
        self.bandit_type = bandit_type
        self.use_rf_style = use_rf_style
        self.use_calibration = use_calibration
        
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
        self.classes_ = None
        self.rng = None
        self.calibrator = None
        
        # OOB tracking
        self.oob_predictions = None  # Shape: (n_samples, n_trees, n_classes)
        self.oob_counts = None       # How many trees have OOB prediction for each sample
        
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
    
    def _stratified_bootstrap(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create stratified bootstrap sample.
        
        Returns:
            (bootstrap_indices, oob_indices)
        """
        n = len(X)
        classes, class_counts = np.unique(y, return_counts=True)
        
        boot_indices = []
        for cls, count in zip(classes, class_counts):
            cls_indices = np.where(y == cls)[0]
            # Sample with replacement, maintaining class proportion
            sampled = rng.choice(cls_indices, size=count, replace=True)
            boot_indices.extend(sampled)
        
        boot_indices = np.array(boot_indices)
        rng.shuffle(boot_indices)
        
        # OOB: samples not in bootstrap
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_indices)] = False
        oob_indices = np.where(oob_mask)[0]
        
        return boot_indices, oob_indices
    
    def _compute_logloss_improvement(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        baseline_prob: float,
    ) -> float:
        """
        Compute log-loss improvement over baseline.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probability for class 1
            baseline_prob: Baseline probability (class prior)
            
        Returns:
            Normalized improvement in [0, 1]
        """
        if len(y_true) == 0:
            return 0.0
        
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
        baseline_prob = np.clip(baseline_prob, eps, 1 - eps)
        
        # Baseline log-loss (predicting class prior for all)
        baseline_proba = np.full_like(y_pred_proba, baseline_prob)
        ll_baseline = log_loss(y_true, baseline_proba, labels=[0, 1])
        
        # Tree log-loss
        ll_tree = log_loss(y_true, y_pred_proba, labels=[0, 1])
        
        # Improvement ratio, clipped to [0, 1]
        if ll_baseline < eps:
            return 0.0
        
        improvement = (ll_baseline - ll_tree) / ll_baseline
        return float(np.clip(improvement, 0.0, 1.0))
    
    def _compute_hardness(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hardness for each sample: |y - p| or negative log prob.
        
        Higher hardness = model was more wrong.
        """
        # Simple: absolute error
        hardness = np.abs(y_true - y_pred_proba)
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
    
    def _compute_disagreement_diversity(
        self,
        predictions: list[np.ndarray],
        buffer_size: int = 20,
    ) -> np.ndarray:
        """
        Compute disagreement-based diversity scores.
        
        div(i,j) = mean(pred_i != pred_j)
        
        Returns diversity score for each tree (lower correlation with others = higher diversity).
        """
        n_trees = len(predictions)
        if n_trees <= 1:
            return np.ones(n_trees)
        
        # Use only last buffer_size trees for efficiency
        start_idx = max(0, n_trees - buffer_size)
        
        diversity_scores = np.zeros(n_trees)
        
        for i in range(n_trees):
            disagreements = []
            # Compare with trees in buffer
            for j in range(start_idx, n_trees):
                if i != j:
                    # Find common samples (both have predictions)
                    mask = ~(np.isnan(predictions[i]) | np.isnan(predictions[j]))
                    if mask.sum() > 0:
                        disagree = np.mean(predictions[i][mask] != predictions[j][mask])
                        disagreements.append(disagree)
            
            if disagreements:
                diversity_scores[i] = np.mean(disagreements)
            else:
                diversity_scores[i] = 0.5  # Neutral diversity
        
        return diversity_scores
    
    def fit(self, train: tuple, val: tuple, seed: int):
        """
        Fit MAPLE v6 with OOB-based evaluation.
        
        Note: val is still passed for API compatibility but not used for tree evaluation.
        Val is only used for final hyperparameter selection if grid searching.
        """
        X_train, y_train = train
        X_val, y_val = val
        
        # Combine train and val for full training (pure bagging uses all data)
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = np.concatenate([y_train, y_val])
        
        assert isinstance(X_full, pd.DataFrame), "X must be DataFrame"
        
        self.classes_ = np.unique(y_full)
        
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
        # Use same seed for all combos to avoid confounding by randomness
        best_score = -np.inf
        best_maple_params = maple_combos[0]
        
        print(f"Grid searching over {len(maple_combos)} MAPLE param combinations...")
        
        for i, maple_params in enumerate(maple_combos):
            self.prior_factory.reset_all_priors()
            
            # Fit and get OOB score - use SAME seed for fair comparison
            self._fit_single(
                X_full, y_full, seed,  # Same seed for all combos
                maple_params=maple_params,
                tree_combos=tree_combos,
            )
            
            # Evaluate using aggregated OOB predictions
            score = self._get_oob_score(y_full)
            
            if score > best_score:
                best_score = score
                best_maple_params = maple_params
        
        print(f"Best MAPLE params: {best_maple_params} with OOB score {best_score:.4f}")
        self.best_params_ = best_maple_params
        
        # Final fit with best params
        self.prior_factory.reset_all_priors()
        self._fit_single(
            X_full, y_full, seed,
            maple_params=best_maple_params,
            tree_combos=tree_combos,
        )
    
    def _get_oob_score(self, y_true: np.ndarray) -> float:
        """Compute balanced accuracy from aggregated OOB predictions."""
        n_samples = len(y_true)
        
        # Aggregate OOB predictions (weighted by tree weights)
        oob_proba = np.zeros((n_samples, len(self.classes_)))
        oob_weight_sum = np.zeros(n_samples)
        
        for t, (tree, feats, weight) in enumerate(zip(self.trees, self.tree_features, self.tree_weights)):
            for i in range(n_samples):
                if self.oob_counts[i, t] > 0:
                    oob_proba[i] += weight * self.oob_predictions[i, t]
                    oob_weight_sum[i] += weight
        
        # Samples with OOB predictions
        valid_mask = oob_weight_sum > 0
        if valid_mask.sum() == 0:
            return 0.0
        
        oob_proba[valid_mask] /= oob_weight_sum[valid_mask, np.newaxis]
        y_pred = self.classes_[np.argmax(oob_proba[valid_mask], axis=1)]
        
        return balanced_accuracy_score(y_true[valid_mask], y_pred)
    
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
        
        # Baseline probability (class prior for class 1)
        baseline_prob = np.mean(y_train == self.classes_[1]) if len(self.classes_) == 2 else 0.5
        
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
        K = len(self.classes_)
        self.oob_predictions = np.zeros((N, n_trees, K), dtype=float)
        self.oob_counts = np.zeros((N, n_trees), dtype=int)
        
        # For diversity calculation
        oob_hard_predictions = []  # List of arrays, each tree's hard predictions (NaN for non-OOB)
        oob_logloss_scores = []
        
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
            
            # 3. Stratified bootstrap
            boot_indices, oob_indices = self._stratified_bootstrap(X_train, y_train, self.rng)
            
            X_boot = X_train.iloc[boot_indices][feat_names_subset]
            y_boot = y_train[boot_indices]
            
            # 4. Select tree params
            tree_params = tree_combos[t % len(tree_combos)]
            
            # 5. Train tree
            tree = DecisionTreeClassifier(
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                class_weight="balanced",
                max_features=max_features_node,
                max_depth=max_depth,
                **tree_params
            )
            tree.fit(X_boot, y_boot)
            
            # 6. OOB evaluation
            if len(oob_indices) > 0:
                X_oob = X_train.iloc[oob_indices][feat_names_subset]
                y_oob = y_train[oob_indices]
                
                # Get probabilities
                oob_proba = tree.predict_proba(X_oob)
                
                # Align with global classes
                aligned_proba = np.zeros((len(oob_indices), K))
                for j, cls in enumerate(tree.classes_):
                    k = int(np.where(self.classes_ == cls)[0][0])
                    aligned_proba[:, k] = oob_proba[:, j]
                
                # Store OOB predictions
                for idx_local, idx_global in enumerate(oob_indices):
                    self.oob_predictions[idx_global, t] = aligned_proba[idx_local]
                    self.oob_counts[idx_global, t] = 1
                
                # Get probability for positive class (for binary)
                if K == 2:
                    pos_class_idx = 1
                    oob_proba_pos = aligned_proba[:, pos_class_idx]
                else:
                    oob_proba_pos = aligned_proba[np.arange(len(y_oob)), y_oob]
                
                # Compute log-loss improvement as reward
                reward = self._compute_logloss_improvement(y_oob, oob_proba_pos, baseline_prob)
                
                # Compute hardness for prior update
                hardness = self._compute_hardness(y_oob, oob_proba_pos)
                
                # OOB log-loss for tree weighting
                oob_ll = log_loss(y_oob, aligned_proba, labels=self.classes_)
                oob_logloss_scores.append(oob_ll)
                
                # Hard predictions for diversity
                oob_hard = np.full(N, np.nan)
                oob_hard[oob_indices] = np.argmax(aligned_proba, axis=1)
                oob_hard_predictions.append(oob_hard)
            else:
                # No OOB samples (rare)
                reward = 0.5
                hardness = np.array([])
                oob_logloss_scores.append(1.0)  # Penalty
                oob_hard_predictions.append(np.full(N, np.nan))
            
            # 7. Store tree
            self.trees.append(tree)
            self.tree_features.append(feat_names_subset)
            self.tree_agents.append(agent_id)
            
            # 8. Update bandit with hardness-scaled reward
            # Trees that perform well on hard samples get higher effective reward
            if len(hardness) > 0:
                mean_hardness = float(np.clip(np.mean(hardness), 0.0, 1.0))
                # Scale reward: good performance on hard samples is more valuable
                # reward_eff in [0, 2*reward] depending on hardness
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
                    reward=reward_eff,  # Use hardness-scaled reward
                    used_features=global_used,
                    is_good_tree=(reward_eff > 0.3),
                    hardness=mean_hardness,  # Pass hardness for tracking
                )
        
        # Compute tree weights based on OOB log-loss and diversity
        oob_logloss_scores = np.array(oob_logloss_scores)
        
        # Diversity scores
        diversity_scores = self._compute_disagreement_diversity(oob_hard_predictions)
        
        # Normalize scores
        ll_range = oob_logloss_scores.max() - oob_logloss_scores.min()
        div_range = diversity_scores.max() - diversity_scores.min()
        
        # Lower log-loss = better, so invert
        if ll_range > 1e-8:
            acc_score = 1 - (oob_logloss_scores - oob_logloss_scores.min()) / ll_range
        else:
            acc_score = np.ones(n_trees)
        
        if div_range > 1e-8:
            div_score = (diversity_scores - diversity_scores.min()) / div_range
        else:
            div_score = np.ones(n_trees)
        
        # Combined score
        combined = (1 - diversity_weight) * acc_score + diversity_weight * div_score
        
        # Softmax weights with temperature
        temperature = 0.5
        weights = np.exp(combined / temperature)
        weights = weights / weights.sum()
        
        # Mix with uniform to avoid collapse
        uniform_weights = np.ones(n_trees) / n_trees
        weights = (1 - weight_uniform_mix) * weights + weight_uniform_mix * uniform_weights
        
        self.tree_weights = weights.tolist()
        
        # Optional: Fit calibrator on OOB predictions
        if self.use_calibration:
            self._fit_calibrator(X_train, y_train)
    
    def _fit_calibrator(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit Platt scaling calibrator using aggregated OOB predictions."""
        N = len(X_train)
        K = len(self.classes_)
        
        # Aggregate OOB predictions
        oob_proba = np.zeros((N, K))
        oob_weight_sum = np.zeros(N)
        
        for t, weight in enumerate(self.tree_weights):
            for i in range(N):
                if self.oob_counts[i, t] > 0:
                    oob_proba[i] += weight * self.oob_predictions[i, t]
                    oob_weight_sum[i] += weight
        
        # Samples with OOB predictions
        valid_mask = oob_weight_sum > 0
        if valid_mask.sum() < 10:
            self.calibrator = None
            return
        
        oob_proba[valid_mask] /= oob_weight_sum[valid_mask, np.newaxis]
        
        # Fit Platt scaling (logistic regression on proba)
        if K == 2:
            # Binary: use log-odds
            X_calib = np.log(oob_proba[valid_mask, 1:] / (oob_proba[valid_mask, :1] + 1e-15))
        else:
            X_calib = oob_proba[valid_mask]
        
        y_calib = y_train[valid_mask]
        
        try:
            self.calibrator = LogisticRegression(max_iter=1000)
            self.calibrator.fit(X_calib, y_calib)
        except Exception:
            self.calibrator = None
    
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
        
        # Apply calibration if available
        if self.calibrator is not None and self.use_calibration:
            if K == 2:
                X_calib = np.log(P[:, 1:] / (P[:, :1] + 1e-15))
                P = self.calibrator.predict_proba(X_calib)
        
        return P