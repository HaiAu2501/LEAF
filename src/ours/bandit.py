"""
Bandit policies for agent allocation in BAPF.
"""
import numpy as np
from abc import ABC, abstractmethod


class BanditPolicy(ABC):
    """Abstract base class for bandit policies."""
    
    def __init__(self, n_arms: int, random_state: int = 0):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(random_state)
        self.reset()
    
    def reset(self):
        """Reset bandit statistics."""
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.rewards = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0
    
    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm to pull."""
        ...
    
    def update(self, arm: int, reward: float) -> None:
        """Update statistics after observing reward."""
        self.counts[arm] += 1
        self.total_pulls += 1
        # Incremental mean update
        n = self.counts[arm]
        self.rewards[arm] += (reward - self.rewards[arm]) / n
    
    def get_mean_rewards(self) -> np.ndarray:
        """Return mean reward estimates for all arms."""
        return self.rewards.copy()


class UCB1(BanditPolicy):
    """
    Upper Confidence Bound (UCB1) bandit policy.
    
    Selects arm with highest UCB = mean_reward + c * sqrt(log(t) / n_arm)
    """
    
    def __init__(self, n_arms: int, exploration_coef: float = 2.0, random_state: int = 0):
        super().__init__(n_arms, random_state)
        self.c = exploration_coef
    
    def select_arm(self) -> int:
        # Initial exploration: pull each arm once
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(self.rng.choice(unpulled))
        
        # UCB selection
        t = self.total_pulls
        ucb_values = self.rewards + self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(ucb_values))


class ThompsonSampling(BanditPolicy):
    """
    Thompson Sampling with Beta prior.
    
    Treats rewards as samples from Bernoulli and maintains Beta posterior.
    """
    
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, random_state: int = 0):
        super().__init__(n_arms, random_state)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alphas = np.full(n_arms, prior_alpha)
        self.betas = np.full(n_arms, prior_beta)
    
    def reset(self):
        super().reset()
        self.alphas = np.full(self.n_arms, self.prior_alpha)
        self.betas = np.full(self.n_arms, self.prior_beta)
    
    def select_arm(self) -> int:
        # Sample from posterior Beta distributions
        samples = self.rng.beta(self.alphas, self.betas)
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Clip reward to [0, 1] for valid Beta update
        r = np.clip(reward, 0.0, 1.0)
        self.alphas[arm] += r
        self.betas[arm] += (1.0 - r)