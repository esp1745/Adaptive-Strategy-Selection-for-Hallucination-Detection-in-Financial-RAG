"""PPO-based adaptive detector selection for hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import math
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .adaptive_selector import AdaptiveDetectorSelector, DetectorPrior


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def contextual_detector_accuracy(
    detector_name: str,
    prior: DetectorPrior,
    state: Dict[str, float],
) -> float:
    """Estimate detector accuracy from benchmark priors plus question signals."""
    numeric_density = state.get("numeric_density", 0.0)
    question_len_norm = min(state.get("question_len", 0.0) / 30.0, 1.0)
    has_percentage = state.get("has_percentage", 0.0)
    has_comparison = state.get("has_comparison", 0.0)
    context_chunks = min(state.get("context_chunks", 0.0) / 3.0, 1.0)
    complexity_score = state.get("complexity_score", 0.0)
    retrieval_confidence = state.get("retrieval_confidence", 0.0)
    score_gap = state.get("score_gap", 0.0)

    adjustment = 0.0
    if detector_name == "token_overlap":
        adjustment += 0.22 * numeric_density
        adjustment += 0.08 * has_percentage
        adjustment += 0.10 * retrieval_confidence
        adjustment -= 0.14 * has_comparison
        adjustment -= 0.08 * question_len_norm
        adjustment -= 0.08 * complexity_score
        adjustment += 0.05 * context_chunks
    elif detector_name == "semantic_similarity":
        adjustment += 0.10 * (1.0 - numeric_density)
        adjustment += 0.03 * question_len_norm
        adjustment += 0.05 * retrieval_confidence
        adjustment -= 0.08 * has_percentage
        adjustment -= 0.05 * has_comparison
    elif detector_name == "bert_nli":
        adjustment += 0.16 * has_comparison
        adjustment += 0.09 * question_len_norm
        adjustment += 0.08 * complexity_score
        adjustment += 0.05 * score_gap
        adjustment -= 0.03 * numeric_density
        adjustment += 0.03 * context_chunks
    elif detector_name == "llm_judge":
        adjustment += 0.24 * has_comparison
        adjustment += 0.16 * question_len_norm
        adjustment += 0.10 * numeric_density
        adjustment += 0.10 * retrieval_confidence
        adjustment += 0.08 * complexity_score
        adjustment += 0.04 * context_chunks
    else:
        adjustment += AdaptiveDetectorSelector.context_bonus_for_detector(detector_name, state)

    return _clamp(prior.accuracy + adjustment, 0.05, 0.99)


def expected_reward(
    correct_probability: float,
    confidence: float,
    cost: float,
    latency_ms: float,
    quality_weight: float = 0.8,
    cost_weight: float = 0.5,
    latency_weight: float = 0.1,
) -> float:
    latency_penalty = min(latency_ms / 10000.0, 1.0)
    quality_signal = float(confidence) * (2.0 * float(correct_probability) - 1.0)
    return quality_weight * quality_signal - cost_weight * float(cost) - latency_weight * latency_penalty


def example_to_state(example: Dict) -> Dict[str, float]:
    raw_context = example.get("context")
    if isinstance(raw_context, list):
        context = [item for item in raw_context if item]
    elif raw_context:
        context = [raw_context]
    else:
        context = []
    answer = example.get("grounded_response") or example.get("response") or ""
    retrieval_stats = {
        "retrieval_confidence": 0.75 if context else 0.35,
        "top_chunk_score": 0.8 if context else 0.0,
        "score_gap": 0.12 if context else 0.0,
    }
    return AdaptiveDetectorSelector.extract_state_features(
        question=example.get("question", ""),
        context=context,
        answer=answer,
        retrieval_stats=retrieval_stats,
    )


def simulated_outcome(
    detector_name: str,
    prior: DetectorPrior,
    state: Dict[str, float],
    rng: random.Random,
) -> Dict[str, float]:
    """Sample a reward from contextual benchmark priors."""
    accuracy_prob = contextual_detector_accuracy(detector_name, prior, state)
    question_len_norm = min(state.get("question_len", 0.0) / 30.0, 1.0)
    latency_mean = prior.avg_latency_ms * (1.0 + 0.08 * question_len_norm)
    latency_std = max(1.0, latency_mean * 0.12)
    latency_ms = max(1.0, rng.gauss(latency_mean, latency_std))
    correct = 1 if rng.random() < accuracy_prob else 0
    confidence = _clamp(accuracy_prob + rng.uniform(-0.05, 0.05), 0.01, 0.99)
    reward = AdaptiveDetectorSelector.compute_reward(
        correct=correct,
        confidence=confidence,
        cost=prior.cost,
        latency_ms=latency_ms,
    )

    return {
        "accuracy_prob": accuracy_prob,
        "correct": correct,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "reward": reward,
    }


@dataclass
class PPOTrainingConfig:
    rollout_size: int = 256
    update_epochs: int = 8
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    hidden_size: int = 32
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, action_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden(x))
        return self.output(hidden)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden(x))
        return self.output(hidden).squeeze(-1)


class PPODetectorSelector:
    """Lightweight PPO policy for detector routing."""

    def __init__(
        self,
        detector_priors: Dict[str, DetectorPrior],
        hidden_size: int = 32,
        feature_mean: Optional[Sequence[float]] = None,
        feature_std: Optional[Sequence[float]] = None,
    ):
        if not detector_priors:
            raise ValueError("detector_priors cannot be empty")

        self.detector_priors = detector_priors
        self.detector_names = list(detector_priors.keys())
        self.feature_names = AdaptiveDetectorSelector.feature_names()
        self.hidden_size = hidden_size
        self.feature_dim = len(self.feature_names)

        self.actor = PolicyNetwork(self.feature_dim, hidden_size, len(self.detector_names))
        self.critic = ValueNetwork(self.feature_dim, hidden_size)

        self.feature_mean = np.array(feature_mean if feature_mean is not None else [0.0] * self.feature_dim, dtype=np.float32)
        self.feature_std = np.array(feature_std if feature_std is not None else [1.0] * self.feature_dim, dtype=np.float32)

    def fit_normalization(self, states: List[Dict[str, float]]) -> None:
        if not states:
            self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
            self.feature_std = np.ones(self.feature_dim, dtype=np.float32)
            return

        matrix = np.array([AdaptiveDetectorSelector.state_to_vector(state) for state in states], dtype=np.float32)
        std = matrix.std(axis=0)
        std[std < 1e-6] = 1.0
        self.feature_mean = matrix.mean(axis=0)
        self.feature_std = std

    def normalize_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.array(vector, dtype=np.float32)
        return (arr - self.feature_mean) / self.feature_std

    def _state_tensor(self, state: Dict[str, float]) -> torch.Tensor:
        vector = AdaptiveDetectorSelector.state_to_vector(state)
        normalized = self.normalize_vector(vector)
        return torch.tensor(normalized, dtype=torch.float32)

    def action_probabilities(self, state: Dict[str, float]) -> Dict[str, float]:
        state_tensor = self._state_tensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return {
            name: float(prob)
            for name, prob in zip(self.detector_names, probs)
        }

    def select_detector(self, state: Dict[str, float], deterministic: bool = True) -> str:
        probabilities = self.action_probabilities(state)
        if deterministic:
            return max(probabilities.items(), key=lambda item: item[1])[0]

        names = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(names, weights=weights, k=1)[0]

    def state_value(self, state: Dict[str, float]) -> float:
        state_tensor = self._state_tensor(state).unsqueeze(0)
        with torch.no_grad():
            return float(self.critic(state_tensor).item())

    def export_policy(self) -> Dict:
        return {
            "type": "mlp_tanh",
            "feature_names": self.feature_names,
            "detectors": self.detector_names,
            "normalization": {
                "mean": self.feature_mean.tolist(),
                "std": self.feature_std.tolist(),
            },
            "actor": {
                "hidden_weight": self.actor.hidden.weight.detach().cpu().tolist(),
                "hidden_bias": self.actor.hidden.bias.detach().cpu().tolist(),
                "output_weight": self.actor.output.weight.detach().cpu().tolist(),
                "output_bias": self.actor.output.bias.detach().cpu().tolist(),
            },
            "critic": {
                "hidden_weight": self.critic.hidden.weight.detach().cpu().tolist(),
                "hidden_bias": self.critic.hidden.bias.detach().cpu().tolist(),
                "output_weight": self.critic.output.weight.detach().cpu().tolist(),
                "output_bias": self.critic.output.bias.detach().cpu().tolist(),
            },
        }

    @classmethod
    def from_policy_payload(
        cls,
        detector_priors: Dict[str, DetectorPrior],
        payload: Dict,
    ) -> "PPODetectorSelector":
        model = payload["policy_model"]
        selector = cls(
            detector_priors=detector_priors,
            hidden_size=len(model["actor"]["hidden_bias"]),
            feature_mean=model["normalization"]["mean"],
            feature_std=model["normalization"]["std"],
        )

        selector.actor.hidden.weight.data.copy_(torch.tensor(model["actor"]["hidden_weight"], dtype=torch.float32))
        selector.actor.hidden.bias.data.copy_(torch.tensor(model["actor"]["hidden_bias"], dtype=torch.float32))
        selector.actor.output.weight.data.copy_(torch.tensor(model["actor"]["output_weight"], dtype=torch.float32))
        selector.actor.output.bias.data.copy_(torch.tensor(model["actor"]["output_bias"], dtype=torch.float32))
        if "critic" in model:
            selector.critic.hidden.weight.data.copy_(torch.tensor(model["critic"]["hidden_weight"], dtype=torch.float32))
            selector.critic.hidden.bias.data.copy_(torch.tensor(model["critic"]["hidden_bias"], dtype=torch.float32))
            selector.critic.output.weight.data.copy_(torch.tensor(model["critic"]["output_weight"], dtype=torch.float32))
            selector.critic.output.bias.data.copy_(torch.tensor(model["critic"]["output_bias"], dtype=torch.float32))
        return selector


def runtime_ppo_update(
    selector: PPODetectorSelector,
    experiences: List[Dict],
    config: PPOTrainingConfig,
) -> Dict[str, float]:
    if not experiences:
        return {"updated": False, "avg_reward": 0.0, "buffer_size": 0}

    optimizer = torch.optim.Adam(
        list(selector.actor.parameters()) + list(selector.critic.parameters()),
        lr=config.learning_rate,
    )

    state_tensor_batch = torch.tensor(
        [experience["state_vector"] for experience in experiences],
        dtype=torch.float32,
    )
    action_tensor_batch = torch.tensor(
        [experience["action_index"] for experience in experiences],
        dtype=torch.long,
    )
    old_log_prob_batch = torch.tensor(
        [experience["old_log_prob"] for experience in experiences],
        dtype=torch.float32,
    )
    reward_batch = torch.tensor(
        [experience["reward"] for experience in experiences],
        dtype=torch.float32,
    )

    with torch.no_grad():
        value_batch = selector.critic(state_tensor_batch)

    advantage_batch = reward_batch - value_batch
    advantage_batch = (advantage_batch - advantage_batch.mean()) / (
        advantage_batch.std(unbiased=False) + 1e-8
    )
    return_batch = reward_batch

    for _ in range(max(1, config.update_epochs // 2)):
        logits = selector.actor(state_tensor_batch)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(action_tensor_batch)
        entropy = dist.entropy().mean()
        values_pred = selector.critic(state_tensor_batch)

        ratios = torch.exp(new_log_probs - old_log_prob_batch)
        unclipped = ratios * advantage_batch
        clipped = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantage_batch
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values_pred, return_batch)
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(selector.actor.parameters()) + list(selector.critic.parameters()),
            config.max_grad_norm,
        )
        optimizer.step()

    return {
        "updated": True,
        "avg_reward": float(reward_batch.mean().item()),
        "buffer_size": float(len(experiences)),
    }


def evaluate_ppo_policy(
    selector: PPODetectorSelector,
    priors: Dict[str, DetectorPrior],
    dataset_examples: List[Dict],
) -> Dict:
    detector_usage = {name: 0 for name in priors.keys()}
    total_accuracy = 0.0
    total_cost = 0.0
    total_latency = 0.0
    total_reward = 0.0
    total_entropy = 0.0

    for example in dataset_examples:
        state = example_to_state(example)
        probabilities = selector.action_probabilities(state)
        chosen = max(probabilities.items(), key=lambda item: item[1])[0]
        prior = priors[chosen]
        accuracy_prob = contextual_detector_accuracy(chosen, prior, state)
        question_len_norm = min(state.get("question_len", 0.0) / 30.0, 1.0)
        latency_ms = prior.avg_latency_ms * (1.0 + 0.08 * question_len_norm)
        reward = expected_reward(
            correct_probability=accuracy_prob,
            confidence=accuracy_prob,
            cost=prior.cost,
            latency_ms=latency_ms,
        )

        detector_usage[chosen] += 1
        total_accuracy += accuracy_prob
        total_cost += prior.cost
        total_latency += latency_ms
        total_reward += reward

        entropy = 0.0
        for probability in probabilities.values():
            if probability > 0:
                entropy -= probability * math.log(probability + 1e-8)
        total_entropy += entropy

    count = max(1, len(dataset_examples))
    return {
        "steps": len(dataset_examples),
        "policy_accuracy": total_accuracy / count,
        "policy_avg_cost": total_cost / count,
        "policy_avg_latency_ms": total_latency / count,
        "policy_avg_reward": total_reward / count,
        "policy_entropy": total_entropy / count,
        "detector_usage": detector_usage,
    }


def contextual_static_baseline(
    priors: Dict[str, DetectorPrior],
    detector_name: str,
    dataset_examples: List[Dict],
) -> Dict[str, float]:
    prior = priors[detector_name]
    total_accuracy = 0.0
    total_latency = 0.0
    total_reward = 0.0

    for example in dataset_examples:
        state = example_to_state(example)
        accuracy_prob = contextual_detector_accuracy(detector_name, prior, state)
        question_len_norm = min(state.get("question_len", 0.0) / 30.0, 1.0)
        latency_ms = prior.avg_latency_ms * (1.0 + 0.08 * question_len_norm)
        total_accuracy += accuracy_prob
        total_latency += latency_ms
        total_reward += expected_reward(
            correct_probability=accuracy_prob,
            confidence=accuracy_prob,
            cost=prior.cost,
            latency_ms=latency_ms,
        )

    count = max(1, len(dataset_examples))
    return {
        "name": detector_name,
        "expected_accuracy": total_accuracy / count,
        "avg_cost": prior.cost,
        "avg_latency_ms": total_latency / count,
        "expected_reward": total_reward / count,
    }


def train_ppo_selector(
    selector: PPODetectorSelector,
    priors: Dict[str, DetectorPrior],
    dataset_examples: List[Dict],
    episodes: int,
    seed: int,
    config: PPOTrainingConfig,
) -> Dict:
    if not dataset_examples:
        raise ValueError("dataset_examples cannot be empty")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    base_examples = list(dataset_examples)
    selector.fit_normalization([example_to_state(example) for example in base_examples])

    training_examples: List[Dict] = []
    for _ in range(episodes):
        shuffled = list(base_examples)
        rng.shuffle(shuffled)
        training_examples.extend(shuffled)

    optimizer = torch.optim.Adam(
        list(selector.actor.parameters()) + list(selector.critic.parameters()),
        lr=config.learning_rate,
    )

    history = []
    action_counts = {name: 0 for name in priors.keys()}

    for update_index, rollout_start in enumerate(range(0, len(training_examples), config.rollout_size), start=1):
        rollout_examples = training_examples[rollout_start:rollout_start + config.rollout_size]
        if not rollout_examples:
            continue

        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        correctness = []
        sampled_latencies = []
        sampled_costs = []

        for example in rollout_examples:
            state = example_to_state(example)
            state_tensor = selector._state_tensor(state)

            with torch.no_grad():
                logits = selector.actor(state_tensor.unsqueeze(0))
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = selector.critic(state_tensor.unsqueeze(0)).squeeze(0)

            action_index = int(action.item())
            detector_name = selector.detector_names[action_index]
            prior = priors[detector_name]
            outcome = simulated_outcome(detector_name, prior, state, rng)

            states.append(state_tensor)
            actions.append(action_index)
            old_log_probs.append(float(log_prob.item()))
            rewards.append(float(outcome["reward"]))
            values.append(float(value.item()))
            correctness.append(int(outcome["correct"]))
            sampled_latencies.append(float(outcome["latency_ms"]))
            sampled_costs.append(prior.cost)
            action_counts[detector_name] += 1

        state_tensor_batch = torch.stack(states)
        action_tensor_batch = torch.tensor(actions, dtype=torch.long)
        old_log_prob_batch = torch.tensor(old_log_probs, dtype=torch.float32)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        value_batch = torch.tensor(values, dtype=torch.float32)
        advantage_batch = reward_batch - value_batch
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std(unbiased=False) + 1e-8)
        return_batch = reward_batch

        for _ in range(config.update_epochs):
            permutation = torch.randperm(len(rollout_examples))
            for batch_start in range(0, len(rollout_examples), config.minibatch_size):
                batch_indices = permutation[batch_start:batch_start + config.minibatch_size]

                batch_states = state_tensor_batch[batch_indices]
                batch_actions = action_tensor_batch[batch_indices]
                batch_old_log_probs = old_log_prob_batch[batch_indices]
                batch_advantages = advantage_batch[batch_indices]
                batch_returns = return_batch[batch_indices]

                logits = selector.actor(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values_pred = selector.critic(batch_states)

                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values_pred, batch_returns)
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(selector.actor.parameters()) + list(selector.critic.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()

        probabilities = []
        with torch.no_grad():
            rollout_logits = selector.actor(state_tensor_batch)
            rollout_probs = torch.softmax(rollout_logits, dim=-1)
            max_probs = rollout_probs.max(dim=-1).values
            probabilities.extend(max_probs.tolist())

        history.append(
            {
                "update": update_index,
                "steps": len(rollout_examples),
                "avg_reward": float(np.mean(rewards)),
                "sample_accuracy": float(np.mean(correctness)),
                "avg_latency_ms": float(np.mean(sampled_latencies)),
                "avg_cost": float(np.mean(sampled_costs)),
                "avg_top_action_prob": float(np.mean(probabilities)),
            }
        )

    return {
        "summary": evaluate_ppo_policy(selector, priors, base_examples),
        "history": history,
        "action_counts": action_counts,
    }
