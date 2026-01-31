"""
Bandit policies for agent allocation in BAPF v5.

Key improvements:
- Smoothed rewards via EMA (Exponential Moving Average)
- Risk-aware selection (mean - std option)
- Rolling window statistics for stable agent evaluation
"""
import numpy as np
from abc import ABC, abstractmethod
from collections import deque


class BanditPolicy(ABC):
    """Abstract base class for bandit policies with smoothed rewards."""
    
    def __init__(
        self,
        n_arms: int,
        random_state: int = 0,
        ema_alpha: float = 0.2,
        window_size: int = 10,
    ):
        """
        Args:
            n_arms: Number of arms (agents)
            random_state: Random seed
            ema_alpha: EMA smoothing factor (higher = more weight on recent)
            window_size: Size of rolling window for statistics
        """
        self.n_arms = n_arms
        self.rng = np.random.default_rng(random_state)
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset bandit statistics."""
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.total_pulls = 0
        
        # EMA-smoothed rewards
        self.ema_rewards = np.zeros(self.n_arms, dtype=float)
        
        # Rolling window for variance estimation
        self.reward_windows: list[deque] = [
            deque(maxlen=self.window_size) for _ in range(self.n_arms)
        ]
        
        # Raw cumulative (for fallback)
        self.cumulative_rewards = np.zeros(self.n_arms, dtype=float)
    
    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm to pull."""
        ...
    
    def update(self, arm: int, reward: float) -> None:
        """
        Update statistics after observing reward.
        
        Uses EMA smoothing instead of simple mean.
        """
        self.counts[arm] += 1
        self.total_pulls += 1
        
        # Cumulative (for mean calculation)
        self.cumulative_rewards[arm] += reward
        
        # EMA update
        if self.counts[arm] == 1:
            # First observation: initialize EMA
            self.ema_rewards[arm] = reward
        else:
            # Exponential moving average
            self.ema_rewards[arm] = (
                self.ema_alpha * reward + 
                (1 - self.ema_alpha) * self.ema_rewards[arm]
            )
        
        # Rolling window
        self.reward_windows[arm].append(reward)
    
    def get_mean_rewards(self) -> np.ndarray:
        """Return EMA reward estimates for all arms."""
        return self.ema_rewards.copy()
    
    def get_raw_mean_rewards(self) -> np.ndarray:
        """Return raw mean rewards (non-smoothed)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            means = self.cumulative_rewards / np.maximum(self.counts, 1)
        return means
    
    def get_reward_std(self) -> np.ndarray:
        """Return standard deviation from rolling window."""
        stds = np.zeros(self.n_arms, dtype=float)
        for i in range(self.n_arms):
            if len(self.reward_windows[i]) >= 2:
                stds[i] = np.std(list(self.reward_windows[i]))
        return stds
    
    def get_risk_adjusted_rewards(self, risk_coef: float = 0.5) -> np.ndarray:
        """
        Return risk-adjusted rewards: mean - risk_coef * std.
        
        Agents with high variance are penalized.
        """
        means = self.ema_rewards
        stds = self.get_reward_std()
        return means - risk_coef * stds


class UCB1(BanditPolicy):
    """
    Upper Confidence Bound (UCB1) bandit policy with smoothed rewards.
    
    Selects arm with highest UCB = ema_reward + c * sqrt(log(t) / n_arm)
    """
    
    def __init__(
        self,
        n_arms: int,
        exploration_coef: float = 2.0,
        random_state: int = 0,
        ema_alpha: float = 0.2,
        window_size: int = 10,
        use_risk_adjusted: bool = False,
        risk_coef: float = 0.3,
    ):
        super().__init__(n_arms, random_state, ema_alpha, window_size)
        self.c = exploration_coef
        self.use_risk_adjusted = use_risk_adjusted
        self.risk_coef = risk_coef
    
    def select_arm(self) -> int:
        # Initial exploration: pull each arm once
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(self.rng.choice(unpulled))
        
        # Base rewards (EMA or risk-adjusted)
        if self.use_risk_adjusted:
            base_rewards = self.get_risk_adjusted_rewards(self.risk_coef)
        else:
            base_rewards = self.ema_rewards
        
        # UCB selection
        t = self.total_pulls
        ucb_values = base_rewards + self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(ucb_values))


class ThompsonSampling(BanditPolicy):
    """
    Thompson Sampling with Beta prior and smoothed updates.
    
    Treats rewards as samples from Bernoulli and maintains Beta posterior.
    Uses EMA-weighted updates for more stable learning.
    """
    
    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        random_state: int = 0,
        ema_alpha: float = 0.2,
        window_size: int = 10,
        decay_factor: float = 0.99,
    ):
        super().__init__(n_arms, random_state, ema_alpha, window_size)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_factor = decay_factor
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
        
        # Decay old observations (forgetting factor for non-stationarity)
        self.alphas *= self.decay_factor
        self.betas *= self.decay_factor
        
        # Ensure minimum values
        self.alphas = np.maximum(self.alphas, self.prior_alpha)
        self.betas = np.maximum(self.betas, self.prior_beta)
        
        # Update with new observation
        self.alphas[arm] += r
        self.betas[arm] += (1.0 - r)


class EpsilonGreedy(BanditPolicy):
    """
    Epsilon-greedy policy with decaying exploration.
    
    Simple but effective baseline.
    """
    
    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        random_state: int = 0,
        ema_alpha: float = 0.2,
        window_size: int = 10,
    ):
        super().__init__(n_arms, random_state, ema_alpha, window_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.current_epsilon = epsilon
    
    def reset(self):
        super().reset()
        self.current_epsilon = self.epsilon
    
    def select_arm(self) -> int:
        # Initial exploration
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(self.rng.choice(unpulled))
        
        # Epsilon-greedy
        if self.rng.random() < self.current_epsilon:
            # Explore
            return int(self.rng.integers(0, self.n_arms))
        else:
            # Exploit
            return int(np.argmax(self.ema_rewards))
    
    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Decay epsilon
        self.current_epsilon = max(
            self.min_epsilon,
            self.current_epsilon * self.epsilon_decay
        )