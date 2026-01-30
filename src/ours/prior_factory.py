"""
Multi-agent Prior Factory for BAPF.

Each agent maintains a prior distribution over features, derived from LLM.
Factory-A approach: sample feature subset according to prior, then train CART.
"""
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
    A single agent that maintains a prior distribution over features.
    
    Each agent can have different:
    - Temperature for LLM sampling
    - System prompt variations
    - Prior weights
    """
    
    def __init__(
        self,
        agent_id: int,
        feature_names: list[str],
        weights: dict[str, float],
        temperature: float = 1.0,
    ):
        self.agent_id = agent_id
        self.feature_names = feature_names
        self.temperature = temperature
        self.d = len(feature_names)
        
        # Convert weights dict to probability distribution
        w = np.array([weights.get(f, 0.5) for f in feature_names], dtype=float)
        w = np.clip(w, 1e-6, None)  # Ensure positive
        self.prior_probs = w / w.sum()
        
        # Statistics
        self.n_trees_generated = 0
        self.total_reward = 0.0
    
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
        return indices.tolist()
    
    def update_stats(self, reward: float):
        """Update agent statistics after generating a tree."""
        self.n_trees_generated += 1
        self.total_reward += reward
    
    @property
    def mean_reward(self) -> float:
        if self.n_trees_generated == 0:
            return 0.0
        return self.total_reward / self.n_trees_generated


class MultiAgentPriorFactory:
    """
    Factory that manages multiple prior agents.
    
    Each agent is created with different LLM parameters (temperature, prompt variations)
    to produce diverse priors over features.
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
    ):
        self.dataset = dataset
        self.model = model
        self.n_agents = min(n_agents, len(self.SYSTEM_PROMPTS))
        self.logger = logger
        self.n_trials = n_trials
        
        self.feature_names = dataset.all_feats
        self.agents: list[PriorAgent] = []
        
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