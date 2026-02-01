import numpy as np
from typing import Optional
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.loader import Dataset
from utils.logger import Logger
from utils.client import LLMClient
from utils.format.weighting import FeatureWeights


DecisionTree = DecisionTreeClassifier | DecisionTreeRegressor


class PriorAgent:
    """
    A single agent that maintains an ADAPTIVE prior distribution over features.
    
    Key improvements:
    - Prior can be updated based on diversity feedback
    - Tracks feature usage for redundancy calculation
    - Supports both tree-level and node-level sampling
    """
    
    def __init__(
        self,
        agent_id: int,
        feature_names: list[str],
        weights: dict[str, float],
        temperature: float = 1.0,
        prior_update_rate: float = 0.1,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent
            feature_names: List of feature names
            weights: Initial weights from LLM
            temperature: LLM temperature used to generate weights
            prior_update_rate: Rate for adaptive prior updates (λ in feedback)
        """
        self.agent_id = agent_id
        self.feature_names = feature_names
        self.temperature = temperature
        self.prior_update_rate = prior_update_rate
        self.d = len(feature_names)
        
        # Store original weights for reference
        self.original_weights = weights.copy()
        
        # Current adaptive weights
        self._weights = np.array(
            [weights.get(f, 0.5) for f in feature_names], dtype=float
        )
        self._weights = np.clip(self._weights, 1e-6, None)
        
        # Track feature usage for redundancy calculation
        self.feature_usage_counts = np.zeros(self.d, dtype=int)
        self.feature_in_good_trees = np.zeros(self.d, dtype=int)  # Features in high-accuracy trees
        
        # Statistics
        self.n_trees_generated = 0
        self.total_reward = 0.0
        self.reward_history: list[float] = []
    
    @property
    def prior_probs(self) -> np.ndarray:
        """Get current probability distribution over features."""
        w = np.clip(self._weights, 1e-6, None)
        return w / w.sum()
    
    def sample_features(self, subset_size: int, rng: np.random.Generator) -> list[int]:
        """
        Sample feature indices according to prior distribution.
        
        Args:
            subset_size: Number of features to sample
            rng: Random number generator
            
        Returns:
            List of feature indices
        """
        subset_size = min(subset_size, self.d)
        indices = rng.choice(
            self.d,
            size=subset_size,
            replace=False,
            p=self.prior_probs
        )
        
        # Track usage
        for idx in indices:
            self.feature_usage_counts[idx] += 1
        
        return indices.tolist()
    
    def get_feature_mask(self, threshold: float = 0.3) -> np.ndarray:
        """
        Get a boolean mask of features to include based on prior weights.
        
        Features with weight < threshold are masked out.
        This enables RF-style node-level feature sampling on a reduced set.
        
        Args:
            threshold: Minimum weight to include feature
            
        Returns:
            Boolean mask of shape (d,)
        """
        return self._weights >= threshold
    
    def update_stats(self, reward: float, used_features: Optional[list[int]] = None):
        """
        Update agent statistics after generating a tree.
        
        Args:
            reward: Validation accuracy (normalized)
            used_features: Feature indices actually used in the tree
        """
        self.n_trees_generated += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        
        # Track which features appear in good trees
        if used_features is not None and reward > 0.6:  # Above median threshold
            for idx in used_features:
                self.feature_in_good_trees[idx] += 1
    
    def update_prior_with_redundancy(
        self,
        global_feature_counts: np.ndarray,
        total_good_trees: int,
    ) -> None:
        """
        Update prior weights based on feature redundancy across all agents.
        
        Features that are overused in good trees across all agents get
        their weight reduced for THIS agent, promoting diversity.
        
        Args:
            global_feature_counts: Count of each feature's appearance in good trees (all agents)
            total_good_trees: Total number of good trees generated
        """
        if total_good_trees == 0:
            return
        
        # Compute redundancy: how often each feature appears in good trees
        redundancy = global_feature_counts / (total_good_trees + 1e-8)
        
        # Update weights: reduce weight for highly redundant features
        # prior_f ← prior_f × exp(-λ × redundancy_f)
        decay = np.exp(-self.prior_update_rate * redundancy)
        self._weights = self._weights * decay
        
        # Ensure minimum weight
        self._weights = np.clip(self._weights, 0.05, 1.0)
        
        # Renormalize to maintain scale
        self._weights = self._weights / self._weights.max()
    
    def reset_prior(self):
        """Reset prior to original LLM-derived weights."""
        self._weights = np.array(
            [self.original_weights.get(f, 0.5) for f in self.feature_names],
            dtype=float
        )
        self._weights = np.clip(self._weights, 1e-6, None)
    
    @property
    def mean_reward(self) -> float:
        if self.n_trees_generated == 0:
            return 0.0
        return self.total_reward / self.n_trees_generated
    
    @property
    def reward_std(self) -> float:
        if len(self.reward_history) < 2:
            return 0.0
        return float(np.std(self.reward_history))


class MultiAgentPriorFactory:
    """
    Factory that manages multiple prior agents with diversity-aware updates.
    
    Key improvements:
    1. Tracks global feature usage for redundancy feedback
    2. Periodically updates agent priors to promote diversity
    3. Supports RF-style feature masking
    """
    
    SYSTEM_PROMPTS = [
        # Agent 0: Balanced/default
        (
            "You are an AutoML prior designer for decision-tree models. "
            "Given dataset context and feature descriptions, propose a weight in [0,1] for each feature. "
            "Higher weight means stronger prior to prefer splits using that feature. "
            "Use the full scale [0,1] when appropriate. Be balanced and consider all aspects."
        ),
        # Agent 1: Domain-focused
        (
            "You are a domain expert designing feature importance priors for decision trees. "
            "Focus on domain knowledge and causal relationships. "
            "Weight features based on their likely causal or predictive relationship with the target. "
            "Strongly prefer features with clear domain justification."
        ),
        # Agent 2: Statistical-focused
        (
            "You are a statistician designing feature priors for decision trees. "
            "Focus on statistical properties: variance, potential for separation, information content. "
            "Prefer features that likely have high information gain or good split potential."
        ),
        # Agent 3: Conservative
        (
            "You are a conservative AutoML designer. "
            "Be cautious with feature weights - only give high weights to features with very strong evidence. "
            "Prefer simpler, more interpretable features. Penalize high-cardinality or noisy features."
        ),
        # Agent 4: Aggressive/exploratory
        (
            "You are an exploratory AutoML designer willing to try unconventional feature combinations. "
            "Don't be afraid to give high weights to features that might seem less obvious. "
            "Look for hidden patterns and non-obvious predictors."
        ),
    ]
    
    TEMPERATURES = [0.7, 0.9, 1.0, 0.8, 1.2]
    
    def __init__(
        self,
        dataset: Dataset,
        model: str = "gpt-4o-mini",
        n_agents: int = 3,
        logger: Optional[Logger] = None,
        n_trials: int = 3,
        prior_update_rate: float = 0.15,
        update_interval: int = 10,
    ):
        """
        Args:
            dataset: Dataset object
            model: LLM model name
            n_agents: Number of agents to create
            logger: Logger for saving outputs
            n_trials: Number of retries for LLM calls
            prior_update_rate: Rate for diversity-based prior updates
            update_interval: How often to update priors (every N trees)
        """
        self.dataset = dataset
        self.model = model
        self.n_agents = min(n_agents, len(self.SYSTEM_PROMPTS))
        self.logger = logger
        self.n_trials = n_trials
        self.prior_update_rate = prior_update_rate
        self.update_interval = update_interval
        
        self.feature_names = dataset.all_feats
        self.d = len(self.feature_names)
        self.agents: list[PriorAgent] = []
        
        # Global tracking for diversity
        self.global_feature_counts = np.zeros(self.d, dtype=int)
        self.total_good_trees = 0
        self.trees_since_update = 0
        
        # Build context message once
        self.human_msg = self._build_message()
        
        # Create agents with different priors
        self._create_agents()
    
    def _build_message(self) -> str:
        """Build the human message with dataset context."""
        msg = f"DATASET: '{self.dataset.name}'\n"
        msg += f"TASK: '{self.dataset.task_type}' with target '{self.dataset.label_col}'.\n\n"
        
        annotations = getattr(self.dataset, 'annotations', {})
        msg += "Feature descriptions:\n"
        for feat in self.dataset.all_feats:
            desc = annotations.get(feat, "No description available.")
            msg += f"- Feature: {feat} | Description: {desc}\n"
        
        return msg
    
    def _get_feature_weights_from_llm(
        self,
        system_prompt: str,
        temperature: float
    ) -> dict[str, float]:
        """Query LLM to get feature weights."""
        client = LLMClient(model=self.model, temperature=temperature)
        
        user_msg = (
            self.human_msg
            + "\nYour task: Assign a weight w in [0,1] to EACH feature.\n"
            + "Weighting rubric:\n"
            + "- 0.90-1.00: Direct proxy/causal driver of the target\n"
            + "- 0.70-0.80: Strong predictor with solid domain rationale\n"
            + "- 0.50-0.60: Moderately informative\n"
            + "- 0.30-0.40: Weak or ambiguous relation\n"
            + "- 0.10-0.20: Marginal relevance\n"
            + "Guidelines:\n"
            + "- Provide exactly one weight for every listed feature\n"
            + "- Weights need not sum to 1\n"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        
        valid_feats = set(self.dataset.all_feats)
        
        for trial in range(self.n_trials):
            try:
                fw: FeatureWeights = client.get_feature_weights(messages=messages)
                
                # Validate coverage
                weights = {}
                for fp in fw.weights:
                    if fp.name in valid_feats:
                        weights[fp.name] = fp.weight
                
                # Check all features covered
                if set(weights.keys()) != valid_feats:
                    missing = valid_feats - set(weights.keys())
                    raise ValueError(f"Missing features: {missing}")
                
                return weights
                
            except Exception as e:
                if trial == self.n_trials - 1:
                    print(f"All trials failed for agent. Error: {e}. Using uniform prior.")
                    return {f: 0.5 for f in self.dataset.all_feats}
                print(f"Trial {trial + 1} failed: {e}. Retrying...")
        
        return {f: 0.5 for f in self.dataset.all_feats}
    
    def _create_agents(self):
        """Create multiple agents with different priors."""
        for i in range(self.n_agents):
            system_prompt = self.SYSTEM_PROMPTS[i]
            temperature = self.TEMPERATURES[i]
            
            print(f"Creating agent {i} with temperature={temperature}...")
            weights = self._get_feature_weights_from_llm(system_prompt, temperature)
            
            agent = PriorAgent(
                agent_id=i,
                feature_names=self.feature_names,
                weights=weights,
                temperature=temperature,
                prior_update_rate=self.prior_update_rate,
            )
            self.agents.append(agent)
            
            if self.logger is not None:
                self.logger.log_to_json(
                    {
                        "agent_id": i,
                        "temperature": temperature,
                        "weights": weights,
                    },
                    f"{self.dataset.name}_agent_{i}_prior.json"
                )
        
        print(f"Created {len(self.agents)} prior agents.")
    
    def get_agent(self, agent_id: int) -> PriorAgent:
        """Get agent by ID."""
        return self.agents[agent_id]
    
    def record_tree_result(
        self,
        agent_id: int,
        reward: float,
        used_features: list[int],
        is_good_tree: bool = False,
    ) -> None:
        """
        Record results from a generated tree and potentially update priors.
        
        Args:
            agent_id: Which agent generated the tree
            reward: Validation accuracy
            used_features: Feature indices used in the tree
            is_good_tree: Whether this tree is considered "good" (high accuracy)
        """
        agent = self.agents[agent_id]
        agent.update_stats(reward, used_features)
        
        # Update global feature counts for good trees
        if is_good_tree:
            for idx in used_features:
                self.global_feature_counts[idx] += 1
            self.total_good_trees += 1
        
        self.trees_since_update += 1
        
        # Periodically update priors for diversity
        if self.trees_since_update >= self.update_interval:
            self._update_all_priors()
            self.trees_since_update = 0
    
    def _update_all_priors(self) -> None:
        """Update all agent priors based on global redundancy."""
        if self.total_good_trees == 0:
            return
        
        # print(f"  Updating priors (global good trees: {self.total_good_trees})...")
        
        for agent in self.agents:
            agent.update_prior_with_redundancy(
                self.global_feature_counts,
                self.total_good_trees,
            )
    
    def get_feature_mask_for_agent(
        self,
        agent_id: int,
        min_features: int = 3,
    ) -> np.ndarray:
        """
        Get feature mask for RF-style node-level sampling.
        
        Returns a boolean mask where True = feature is available for splitting.
        This allows using DecisionTree with max_features on a reduced set.
        
        Args:
            agent_id: Agent ID
            min_features: Minimum number of features to include
            
        Returns:
            Boolean mask of shape (d,)
        """
        agent = self.agents[agent_id]
        probs = agent.prior_probs
        
        # Include top features by probability
        # At minimum, include min_features
        n_include = max(min_features, int(self.d * 0.5))
        
        # Get indices of top features
        top_indices = np.argsort(probs)[-n_include:]
        
        mask = np.zeros(self.d, dtype=bool)
        mask[top_indices] = True
        
        return mask
    
    def reset_all_priors(self) -> None:
        """Reset all agents to their original LLM-derived priors."""
        for agent in self.agents:
            agent.reset_prior()
        
        # Reset global tracking
        self.global_feature_counts = np.zeros(self.d, dtype=int)
        self.total_good_trees = 0
        self.trees_since_update = 0
    
    def sample_tree(
        self,
        agent_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        subset_size: int,
        tree_params: dict,
        rng: np.random.Generator,
        task_type: str = "classification",
    ) -> tuple[DecisionTree, list[int]]:
        """
        Sample a tree from the specified agent's prior.
        
        Args:
            agent_id: Which agent to use
            X_train: Training features (DataFrame)
            y_train: Training targets
            subset_size: Number of features to use
            tree_params: Parameters for DecisionTree
            rng: Random number generator
            task_type: "classification" or "regression"
            
        Returns:
            (fitted_tree, feature_indices)
        """
        agent = self.agents[agent_id]
        
        # Sample feature subset according to agent's prior
        feat_indices = agent.sample_features(subset_size, rng)
        
        # Get feature subset
        X_subset = X_train.iloc[:, feat_indices] if hasattr(X_train, 'iloc') else X_train[:, feat_indices]
        
        # Create and fit tree
        if task_type == "classification":
            tree = DecisionTreeClassifier(
                random_state=int(rng.integers(0, 2**31 - 1)),
                class_weight="balanced",
                **tree_params
            )
        else:
            tree = DecisionTreeRegressor(
                random_state=int(rng.integers(0, 2**31 - 1)),
                **tree_params
            )
        
        tree.fit(X_subset, y_train)
        
        return tree, feat_indices