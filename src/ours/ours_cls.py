"""
Bandit-Allocated Prior Factories (BAPF) Classifier v4.

Pure Bagging approach (like Random Forest):
- All trees trained independently in parallel
- No sequential dependency on ensemble
- Diversity computed post-hoc for weighting
"""
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

from src.template import Algorithm
from src.ours.prior_factory import MultiAgentPriorFactory
from src.ours.bandit import UCB1, ThompsonSampling
from utils.logger import Logger


class BAPFClassifier(Algorithm):
    """
    Bandit-Allocated Prior Factories Classifier - Bagging Version.
    
    Like Random Forest but with LLM-guided feature sampling:
    - Each tree is independent (bagging)
    - Bandit learns which agent produces better trees
    - Final prediction: weighted majority vote
    """
    
    def __init__(
        self,
        logger: Logger,
        param_grid: dict[str, list] = None,
        n_agents: int = 3,
        exploration_coef: float = 0.5,
        feature_subset_ratio: float = 0.7,
        bandit_type: str = "ucb",
    ):
        super().__init__(
            logger=logger,
            name="BAPF",
            task_type="classification",
            param_grid=param_grid,
        )
        self.n_agents = n_agents
        self.exploration_coef = exploration_coef
        self.feature_subset_ratio = feature_subset_ratio
        self.bandit_type = bandit_type
        
        self.prior_factory = None
        self.bandit = None
        self.trees = []
        self.tree_features = []
        self.tree_weights = []
        self.tree_agents = []
        self.classes_ = None
        self.rng = None
    
    def setup(self, dataset, model):
        """Initialize multi-agent prior factory."""
        self.prior_factory = MultiAgentPriorFactory(
            dataset=dataset,
            model=model,
            n_agents=self.n_agents,
            logger=self.logger,
        )
    
    def fit(self, train: tuple, val: tuple, seed: int):
        """
        Fit BAPF ensemble using Bagging.
        
        Phase 1: Train all trees independently (bagging)
        Phase 2: Compute diversity-aware weights
        """
        self.trees = []
        self.tree_features = []
        self.tree_weights = []
        self.tree_agents = []
        self.rng = np.random.default_rng(seed)
        
        X_train, y_train = train
        X_val, y_val = val
        
        assert isinstance(X_train, pd.DataFrame), "X_train must be DataFrame"
        assert isinstance(X_val, pd.DataFrame), "X_val must be DataFrame"
        
        self.classes_ = np.unique(y_train)
        feature_names = list(X_train.columns)
        d = len(feature_names)
        N = len(X_train)
        n_val = len(X_val)
        
        # Ensemble size
        n_estimators_list = self.param_grid.get("n_estimators", [50])
        n_trees = int(max(n_estimators_list))
        
        # Feature subset size
        m_sub = max(1, int(d * self.feature_subset_ratio))
        
        # Tree hyperparameter grid
        grid_no_n = {k: v for k, v in self.param_grid.items() if k != "n_estimators"}
        if len(grid_no_n) == 0:
            tree_param_combos = [{}]
        else:
            keys = list(grid_no_n.keys())
            vals = [grid_no_n[k] for k in keys]
            tree_param_combos = [dict(zip(keys, prod)) for prod in itertools.product(*vals)]
        
        # Initialize bandit
        if self.bandit_type == "ts":
            self.bandit = ThompsonSampling(n_arms=self.n_agents, random_state=seed)
        else:
            self.bandit = UCB1(
                n_arms=self.n_agents, 
                exploration_coef=self.exploration_coef,
                random_state=seed
            )
        
        # ============ PHASE 1: Train all trees independently (Bagging) ============
        val_accuracies = []
        val_predictions = []  # Store predictions for diversity calculation
        
        for i in range(n_trees):
            # 1. Bandit selects agent
            agent_id = self.bandit.select_arm()
            agent = self.prior_factory.get_agent(agent_id)
            
            # 2. Sample features according to agent's prior
            feat_indices = agent.sample_features(m_sub, self.rng)
            feat_names_subset = [feature_names[j] for j in feat_indices]
            
            # 3. Bootstrap sample (key part of bagging)
            boot_idx = self.rng.integers(0, N, size=N)
            X_boot = X_train.iloc[boot_idx][feat_names_subset]
            y_boot = y_train[boot_idx]
            
            # 4. Find best tree params via validation
            best_tree = None
            best_val_acc = -1
            best_val_pred = None
            
            for params in tree_param_combos:
                tree = DecisionTreeClassifier(
                    random_state=int(self.rng.integers(0, 2**31 - 1)),
                    class_weight="balanced",
                    **params
                )
                tree.fit(X_boot, y_boot)
                
                y_pred_val = tree.predict(X_val[feat_names_subset])
                val_acc = balanced_accuracy_score(y_val, y_pred_val)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_tree = tree
                    best_val_pred = y_pred_val
            
            # 5. Store tree (independent of other trees - bagging)
            self.trees.append(best_tree)
            self.tree_features.append(feat_names_subset)
            self.tree_agents.append(agent_id)
            val_accuracies.append(best_val_acc)
            val_predictions.append(best_val_pred)
            
            # 6. Update bandit with accuracy as reward
            reward = (best_val_acc - 0.5) * 2  # Normalize to [0, 1]
            reward = np.clip(reward, 0.0, 1.0)
            self.bandit.update(agent_id, reward)
            agent.update_stats(reward)
        
        # ============ PHASE 2: Compute diversity-aware weights ============
        val_accuracies = np.array(val_accuracies)
        val_predictions = np.array(val_predictions)  # Shape: (n_trees, n_val)
        
        # Compute pairwise disagreement matrix
        n_trees_actual = len(self.trees)
        disagreement = np.zeros((n_trees_actual, n_trees_actual))
        for i in range(n_trees_actual):
            for j in range(i + 1, n_trees_actual):
                dis = np.mean(val_predictions[i] != val_predictions[j])
                disagreement[i, j] = dis
                disagreement[j, i] = dis
        
        # Diversity score: average disagreement with other trees
        diversity_scores = disagreement.sum(axis=1) / (n_trees_actual - 1 + 1e-8)
        
        # Combined score: accuracy + diversity bonus
        # Normalize both to [0, 1]
        acc_norm = (val_accuracies - val_accuracies.min()) / (val_accuracies.max() - val_accuracies.min() + 1e-8)
        div_norm = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        
        # Weight diversity less than accuracy (accuracy is primary)
        combined = 0.8 * acc_norm + 0.2 * div_norm
        
        # Convert to weights via softmax
        temperature = 0.5
        weights = np.exp(combined / temperature)
        weights = weights / weights.sum()
        
        self.tree_weights = weights.tolist()
        
        # ============ Logging ============
        agent_counts = [0] * self.n_agents
        agent_accs = [[] for _ in range(self.n_agents)]
        for a, acc in zip(self.tree_agents, val_accuracies):
            agent_counts[a] += 1
            agent_accs[a].append(acc)
        
        print(f"BAPF (Bagging): {n_trees} trees, {self.n_agents} agents")
        print(f"  Trees per agent: {agent_counts}")
        print(f"  Mean val acc per agent: {[f'{np.mean(r):.4f}' if r else 'N/A' for r in agent_accs]}")
        print(f"  Overall mean val acc: {val_accuracies.mean():.4f}")
        print(f"  Mean diversity: {diversity_scores.mean():.4f}")
    
    def predict(self, X_test) -> np.ndarray:
        """Predict class labels via weighted majority vote."""
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X_test) -> np.ndarray:
        """Predict class probabilities using weighted voting."""
        if len(self.trees) == 0:
            raise ValueError("Model not fitted.")
        
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        K = len(self.classes_)
        n_samples = len(X_test_df)
        
        P = np.zeros((n_samples, K), dtype=float)
        
        for tree, feats, weight in zip(self.trees, self.tree_features, self.tree_weights):
            proba = tree.predict_proba(X_test_df[feats])
            
            # Align to global classes
            aligned = np.zeros((n_samples, K), dtype=float)
            for j, cls in enumerate(tree.classes_):
                k = int(np.where(self.classes_ == cls)[0][0])
                aligned[:, k] = proba[:, j]
            
            P += weight * aligned
        
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        
        return P