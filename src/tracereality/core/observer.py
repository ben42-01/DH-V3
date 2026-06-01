"""Neural observer agent.

Each observer is a small neural network that learns to predict
next states from observed trace history. Supports mutation,
cloning, and weight perturbation for evolutionary dynamics.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorNet(nn.Module):
    """Small MLP with embedding layer for next-state prediction and optional policy head.

    Architecture:
        Embedding(n_states) -> Linear(embed_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, hidden_dim) -> ReLU ->
        -- prediction_head: Linear(hidden_dim, n_states)
        -- action_head (optional): Linear(hidden_dim, n_actions)
    """

    def __init__(
        self,
        n_states: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        history_window_size: int = 1,
        n_actions: int = 0,  # 0 = no policy learning
    ):
        super().__init__()
        self.n_states = n_states
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.history_window_size = history_window_size
        self.n_actions = n_actions

        # Embedding: one for each state (used per position in window)
        self.embedding = nn.Embedding(n_states, embedding_dim)

        # Input size depends on whether we use history window
        input_size = embedding_dim * history_window_size

        # Build shared MLP layers (without final output)
        layers = []
        prev_dim = input_size
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared_mlp = nn.Sequential(*layers)

        # Prediction head (next-state prediction)
        self.prediction_head = nn.Linear(hidden_dim, n_states)

        # Action head (policy) — only created when n_actions > 0
        self.action_head: nn.Linear | None = None
        if n_actions > 0:
            self.action_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size,) for single state
               or (batch_size, history_window_size) for history window.

        Returns:
            If action_head is None:
                Prediction logits of shape (batch_size, n_states).
            Else:
                Tuple of (prediction_logits, action_logits).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            x = self.embedding(x)
            x = x.view(x.size(0), -1)
        else:
            x = self.embedding(x)
            x = x.view(x.size(0), -1)

        hidden = self.shared_mlp(x)
        pred_logits = self.prediction_head(hidden)

        if self.action_head is not None:
            action_logits = self.action_head(hidden)
            return pred_logits, action_logits

        return pred_logits

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted next-state probability distribution.

        Args:
            x: Input tensor.

        Returns:
            Probability distribution of shape (batch_size, n_states).
        """
        if self.action_head is not None:
            pred_logits, _ = self.forward(x)
        else:
            pred_logits = self.forward(x)
        return F.softmax(pred_logits, dim=-1)

    def predict_action_probs(self, x: torch.Tensor) -> torch.Tensor | None:
        """Get predicted action probability distribution (policy).

        Args:
            x: Input tensor.

        Returns:
            Action probability distribution of shape (batch_size, n_actions),
            or None if action_head is not configured.
        """
        if self.action_head is None:
            return None
        if self.action_head is not None:
            _, action_logits = self.forward(x)
            return F.softmax(action_logits, dim=-1)
        return None

    def get_transition_matrix_estimate(
        self,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Estimate the full transition matrix by feeding each state.

        For each state i, computes P(next | current=i).
        If the model uses a history window, the window is filled
        with the same state (repeated).

        Args:
            device: Torch device.

        Returns:
            Estimated transition matrix of shape (n_states, n_states).
        """
        if device is None:
            device = next(self.parameters()).device

        P_est = np.zeros((self.n_states, self.n_states))
        with torch.no_grad():
            for i in range(self.n_states):
                # Build proper input: if history window is used, repeat the state
                if self.history_window_size > 1:
                    x = torch.full((1, self.history_window_size), i, dtype=torch.long, device=device)
                else:
                    x = torch.tensor([i], device=device)
                probs = self.predict_probs(x).cpu().numpy()[0]
                P_est[i] = probs
        return P_est


class NeuralObserver:
    """An observer agent with a neural predictor model.

    Each observer has:
    - A neural network model for next-state prediction (and optional policy head)
    - An optimizer (Adam)
    - A mutation rate for evolutionary perturbations
    - Fitness history tracking
    - Model parameter snapshots for stability analysis
    - Optional policy learning (actions, rewards, cumulative reward)
    """

    def __init__(
        self,
        observer_id: int,
        n_states: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        history_window_size: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        mutation_rate: float = 0.01,
        device: str | torch.device | None = None,
        n_actions: int = 0,
    ):
        """Initialize a neural observer.

        Args:
            observer_id: Unique identifier.
            n_states: Number of states in the world.
            embedding_dim: Dimension of state embeddings.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of MLP hidden layers.
            history_window_size: Number of past states to use as input.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            mutation_rate: Rate for weight perturbations during evolution.
            device: Torch device ('cpu' or 'cuda').
            n_actions: Number of actions for policy learning (0 = no policy).
        """
        self.id = observer_id
        self.n_states = n_states
        self.history_window_size = history_window_size
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.n_actions = n_actions

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = PredictorNet(
            n_states=n_states,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            history_window_size=history_window_size,
            n_actions=n_actions,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Tracking
        self.fitness_history: List[float] = []
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.param_snapshots: List[np.ndarray] = []
        self._step_count = 0

        # Running accuracy counters for O(1) incremental accuracy computation
        self._cumulative_correct: int = 0
        self._cumulative_total: int = 0

        # Cached transition matrix: recomputed only when dirty
        self._cached_P: np.ndarray | None = None
        self._P_dirty: bool = True

        # Policy learning fields (only used when n_actions > 0)
        self.cumulative_reward: float = 0.0
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        # Running window for policy gradient update
        self._recent_rewards: List[float] = []
        self._recent_actions: List[int] = []
        self._recent_states: List[int] = []

    def predict(self, state_or_window: np.ndarray) -> np.ndarray:
        """Predict next-state distribution from input.

        Args:
            state_or_window: Either a single state (scalar) or
                history window (array of length window_size).

        Returns:
            Predicted distribution of shape (n_states,).
        """
        self.model.eval()
        x = torch.as_tensor(state_or_window, dtype=torch.long, device=self.device)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        with torch.no_grad():
            probs = self.model.predict_probs(x)
        return probs.cpu().numpy()

    def compute_loss(
        self,
        predicted_probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            predicted_probs: Predicted distributions of shape (batch, n_states).
            target: True next states of shape (batch,).

        Returns:
            Cross-entropy loss (scalar tensor).
        """
        return F.cross_entropy(predicted_probs, target)

    def train_step(
        self,
        input_states: np.ndarray | torch.Tensor,
        target_states: np.ndarray | torch.Tensor,
    ) -> float:
        """Perform one training step.

        Args:
            input_states: Input states (batch_size,) or (batch_size, window_size).
            target_states: Target next states (batch_size,).

        Returns:
            Loss value (float).
        """
        self.model.train()

        x = torch.as_tensor(input_states, dtype=torch.long, device=self.device)
        y = torch.as_tensor(target_states, dtype=torch.long, device=self.device)

        self.optimizer.zero_grad()
        output = self.model(x)
        if isinstance(output, tuple):
            pred_logits, _ = output
        else:
            pred_logits = output
        loss = F.cross_entropy(pred_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)

        # Compute accuracy and update running counters
        preds = pred_logits.argmax(dim=-1)
        correct = (preds == y).sum().item()
        self._cumulative_correct += correct
        self._cumulative_total += len(y)
        acc = correct / len(y)
        self.accuracy_history.append(acc)

        # Mark cached transition matrix as stale
        self._P_dirty = True

        self._step_count += 1

        return loss_val

    @property
    def running_accuracy(self) -> float:
        """O(1) cumulative accuracy across all training steps."""
        if self._cumulative_total == 0:
            return 0.0
        return self._cumulative_correct / self._cumulative_total

    def mutate(self, magnitude: float | None = None) -> None:
        """Perturb model weights by Gaussian noise.

        Adds noise to all parameters: param += noise * mutation_rate * magnitude.

        Args:
            magnitude: Scale factor for mutation. If None, uses self.mutation_rate.
        """
        rate = magnitude if magnitude is not None else self.mutation_rate
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * rate
                param.add_(noise)
        self._P_dirty = True

    def set_mutation_rate(self, rate: float) -> None:
        """Update the mutation rate."""
        self.mutation_rate = rate

    def clone(self, new_id: int) -> "NeuralObserver":
        """Create a deep copy of this observer with a new ID.

        Args:
            new_id: ID for the clone.

        Returns:
            A new NeuralObserver with copied weights and optimizer state.
        """
        clone = NeuralObserver(
            observer_id=new_id,
            n_states=self.n_states,
            embedding_dim=self.model.embedding_dim,
            hidden_dim=self.model.hidden_dim,
            n_layers=self.model.n_layers,
            history_window_size=self.history_window_size,
            learning_rate=self.learning_rate,
            weight_decay=0.0,  # Don't carry over specific settings
            mutation_rate=self.mutation_rate,
            device=self.device,
            n_actions=self.n_actions,
        )
        # Copy model weights
        clone.model.load_state_dict(self.model.state_dict())
        # Copy optimizer state
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        # Clone running accuracy counters
        clone._cumulative_correct = self._cumulative_correct
        clone._cumulative_total = self._cumulative_total
        return clone

    def snapshot_params(self) -> None:
        """Save current model parameters as a flattened vector (for stability analysis)."""
        params = []
        with torch.no_grad():
            for p in self.model.parameters():
                params.append(p.cpu().numpy().ravel())
        snapshot = np.concatenate(params)
        self.param_snapshots.append(snapshot)

    def get_param_variance(self, n_recent: int = 10) -> float:
        """Compute variance of parameters over recent snapshots.

        Low variance = stable model.

        Args:
            n_recent: Number of recent snapshots to use.

        Returns:
            Mean variance across all parameters.
        """
        if len(self.param_snapshots) < 2:
            return 0.0

        recent = self.param_snapshots[-n_recent:]
        stacked = np.stack(recent)
        variance = np.var(stacked, axis=0).mean()
        return float(variance)

    def get_transition_matrix_estimate(self) -> np.ndarray:
        """Get the observer's estimated transition matrix.

        Uses cached version if the model hasn't changed since last compute.
        Recomputes only when weights have been modified (train_step or mutate).

        Returns:
            Matrix of shape (n_states, n_states).
        """
        if self._cached_P is None or self._P_dirty:
            self._cached_P = self.model.get_transition_matrix_estimate(self.device)
            self._P_dirty = False
        return self._cached_P

    def record_fitness(self, fitness: float) -> None:
        """Record a fitness value."""
        self.fitness_history.append(fitness)

    @property
    def current_fitness(self) -> float:
        """Most recent fitness value, or 0 if none."""
        return self.fitness_history[-1] if self.fitness_history else 0.0

    @property
    def average_accuracy(self) -> float:
        """Average accuracy over all training steps.

        Uses running counters for O(1) computation.
        """
        if self._cumulative_total == 0:
            return 0.0
        return self._cumulative_correct / self._cumulative_total

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        """Get serializable state."""
        return {
            "id": self.id,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "fitness_history": self.fitness_history,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "mutation_rate": self.mutation_rate,
            "n_states": self.n_states,
            "cumulative_correct": self._cumulative_correct,
            "cumulative_total": self._cumulative_total,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load serialized state."""
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.fitness_history = state["fitness_history"]
        self.loss_history = state["loss_history"]
        self.accuracy_history = state["accuracy_history"]
        self.mutation_rate = state["mutation_rate"]
        self._cumulative_correct = state.get("cumulative_correct", 0)
        self._cumulative_total = state.get("cumulative_total", 0)