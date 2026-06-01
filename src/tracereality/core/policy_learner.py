"""Policy gradient (REINFORCE) learner for utility-bearing observers.

Trains the action head of observer neural networks using
the REINFORCE algorithm with discounted rewards.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .observer import NeuralObserver


class PolicyGradientLearner:
    """REINFORCE-style policy gradient trainer.

    Observers accumulate (state, action, reward) trajectories.
    At each update step, the policy is updated to increase
    the probability of actions that led to higher cumulative reward.
    """

    def __init__(
        self,
        discount_factor: float = 0.99,
        policy_learning_rate: float = 1e-3,
        update_interval: int = 100,
    ):
        """Initialize the policy gradient learner.

        Args:
            discount_factor: Discount factor for future rewards (gamma).
            policy_learning_rate: Learning rate for policy updates.
            update_interval: Steps between policy gradient updates.
        """
        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.update_interval = update_interval
        self._step_counter = 0

    def record_transition(
        self,
        observer: NeuralObserver,
        state: int,
        action: int,
        reward: float,
    ) -> None:
        """Record a (state, action, reward) transition for an observer.

        Args:
            observer: The observer that took the action.
            state: The state before the action.
            action: The action taken.
            reward: The reward received.
        """
        observer._recent_states.append(state)
        observer._recent_actions.append(action)
        observer._recent_rewards.append(reward)
        observer.action_history.append(action)
        observer.reward_history.append(reward)
        observer.cumulative_reward += reward

    def maybe_update(
        self,
        observer: NeuralObserver,
        force: bool = False,
    ) -> float:
        """Perform a policy gradient update if enough steps have accumulated.

        Uses the REINFORCE algorithm:
            ∇θ J ≈ Σ_t ∇θ log π(a_t | s_t) · G_t
        where G_t is the discounted return from time t.

        Args:
            observer: The observer to update.
            force: If True, update even without enough recent data.

        Returns:
            Policy loss value, or 0.0 if no update performed.
        """
        self._step_counter += 1

        if not force and self._step_counter % self.update_interval != 0:
            return 0.0

        if len(observer._recent_states) < 2:
            return 0.0

        # Compute discounted returns
        rewards = np.array(observer._recent_rewards, dtype=np.float32)
        returns = self._compute_discounted_returns(rewards)

        # Standardize returns for stability
        if len(returns) > 1 and np.std(returns) > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Perform REINFORCE update
        loss = self._reinforce_update(
            observer,
            np.array(observer._recent_states, dtype=np.int64),
            np.array(observer._recent_actions, dtype=np.int64),
            returns,
        )

        # Clear recent buffers
        observer._recent_states.clear()
        observer._recent_actions.clear()
        observer._recent_rewards.clear()

        return loss

    def _compute_discounted_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns G_t for each timestep.

        Args:
            rewards: Array of rewards, shape (T,).

        Returns:
            Array of discounted returns, shape (T,).
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.discount_factor * running_return
            returns[t] = running_return
        return returns

    def _reinforce_update(
        self,
        observer: NeuralObserver,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Perform a REINFORCE policy gradient update.

        Args:
            observer: The observer to update.
            states: Array of states, shape (T,).
            actions: Array of actions, shape (T,).
            returns: Array of discounted returns, shape (T,).

        Returns:
            Loss value.
        """
        if observer.model.action_head is None:
            return 0.0

        observer.model.train()
        observer.optimizer.zero_grad()

        # Build proper input: if history window is used, repeat states across window
        window_size = observer.history_window_size
        if window_size > 1:
            x = torch.as_tensor(states, dtype=torch.long, device=observer.device)
            x = x.unsqueeze(-1).expand(-1, window_size)  # (T,) -> (T, window_size)
        else:
            x = torch.as_tensor(states, dtype=torch.long, device=observer.device)
        _, action_logits = observer.model(x)

        # Compute log probabilities of taken actions (REINFORCE loss)
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_indices = torch.as_tensor(actions, dtype=torch.long, device=observer.device)
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        # REINFORCE: loss = - Σ_t log π(a_t|s_t) · G_t
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=observer.device)
        loss = -(selected_log_probs * returns_tensor).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(observer.model.parameters(), max_norm=1.0)
        observer.optimizer.step()

        # Mark transition matrix as stale
        observer._P_dirty = True

        return loss.item()