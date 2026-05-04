"""Train an adaptive detector selector with either a bandit or PPO policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch

from benchmark_detectors import load_test_dataset
from src.rl import (
    AdaptiveDetectorSelector,
    DetectorPrior,
    PPODetectorSelector,
    PPOTrainingConfig,
    contextual_static_baseline,
    train_ppo_selector,
)


def load_benchmark_priors(path: Path) -> Dict[str, DetectorPrior]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    priors: Dict[str, DetectorPrior] = {}
    for name, entry in raw.items():
        priors[name] = DetectorPrior(
            name=name,
            accuracy=float(entry["accuracy"]),
            avg_latency_ms=float(entry["avg_latency_ms"]),
            cost=float(entry["cost"]),
        )
    return priors


def simulate_bandit_training(
    selector: AdaptiveDetectorSelector,
    priors: Dict[str, DetectorPrior],
    episodes: int,
    dataset_examples: List[Dict],
    seed: int,
) -> Dict:
    random.seed(seed)

    total_steps = 0
    policy_correct = 0
    policy_cost = 0.0
    policy_latency = 0.0
    detector_usage = {name: 0 for name in priors.keys()}

    for _ in range(episodes):
        for example in dataset_examples:
            state = selector.extract_state_features(
                question=example.get("question", ""),
                context=[example.get("context", "")] if example.get("context") else [],
            )
            chosen = selector.select_detector(state)
            prior = priors[chosen]

            correct = 1 if random.random() < prior.accuracy else 0
            latency = max(1.0, random.gauss(prior.avg_latency_ms, max(1.0, prior.avg_latency_ms * 0.15)))
            confidence = min(max(prior.accuracy + random.uniform(-0.08, 0.08), 0.01), 0.99)

            reward = selector.compute_reward(
                correct=correct,
                confidence=confidence,
                cost=prior.cost,
                latency_ms=latency,
            )
            selector.update(chosen, reward=reward, correct=correct)

            total_steps += 1
            policy_correct += correct
            policy_cost += prior.cost
            policy_latency += latency
            detector_usage[chosen] += 1

    return {
        "steps": total_steps,
        "policy_accuracy": policy_correct / max(1, total_steps),
        "policy_avg_cost": policy_cost / max(1, total_steps),
        "policy_avg_latency_ms": policy_latency / max(1, total_steps),
        "detector_usage": detector_usage,
    }


def expected_static_baseline(priors: Dict[str, DetectorPrior], detector_name: str) -> Dict[str, float]:
    detector = priors[detector_name]
    return {
        "name": detector_name,
        "expected_accuracy": detector.accuracy,
        "avg_cost": detector.cost,
        "avg_latency_ms": detector.avg_latency_ms,
    }


def save_policy(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(summary: Dict, baselines: Dict) -> None:
    print("\n=== RL TRAINING SUMMARY ===")
    print(f"Steps: {summary['steps']}")
    print(f"Policy accuracy: {summary['policy_accuracy']:.3f}")
    print(f"Policy avg cost: {summary['policy_avg_cost']:.3f}")
    print(f"Policy avg latency: {summary['policy_avg_latency_ms']:.1f}ms")
    if "policy_avg_reward" in summary:
        print(f"Policy avg reward: {summary['policy_avg_reward']:.3f}")
    if "policy_entropy" in summary:
        print(f"Policy entropy: {summary['policy_entropy']:.3f}")

    print("Detector usage:")
    for name, count in summary["detector_usage"].items():
        print(f"  - {name}: {count}")

    print("\nStatic baselines:")
    for key, baseline in baselines.items():
        line = (
            f"  - {key}: acc={baseline['expected_accuracy']:.3f}, "
            f"cost={baseline['avg_cost']:.3f}, latency={baseline['avg_latency_ms']:.1f}ms"
        )
        if "expected_reward" in baseline:
            line += f", reward={baseline['expected_reward']:.3f}"
        print(line)


def build_bandit_payload(args, priors: Dict[str, DetectorPrior], examples: List[Dict]) -> Dict:
    selector = AdaptiveDetectorSelector(
        detector_priors=priors,
        epsilon=args.epsilon,
        prior_strength=args.prior_strength,
    )

    print("Training RL selector with bandit simulation...")
    summary = simulate_bandit_training(
        selector=selector,
        priors=priors,
        episodes=args.episodes,
        dataset_examples=examples,
        seed=args.seed,
    )

    best_accuracy_detector = max(priors.values(), key=lambda prior: prior.accuracy).name
    cheapest_detector = min(priors.values(), key=lambda prior: prior.cost).name
    llm_detector = "llm_judge" if "llm_judge" in priors else best_accuracy_detector

    baselines = {
        "best_accuracy": expected_static_baseline(priors, best_accuracy_detector),
        "cheapest": expected_static_baseline(priors, cheapest_detector),
        "always_llm": expected_static_baseline(priors, llm_detector),
    }

    payload = {
        "algorithm": "bandit",
        "config": {
            "algorithm": "bandit",
            "episodes": args.episodes,
            "limit": args.limit,
            "epsilon": args.epsilon,
            "prior_strength": args.prior_strength,
            "seed": args.seed,
        },
        "summary": summary,
        "baselines": baselines,
        "policy": selector.policy_snapshot(),
    }
    return payload


def build_ppo_payload(args, priors: Dict[str, DetectorPrior], examples: List[Dict]) -> Dict:
    torch.manual_seed(args.seed)
    training_config = PPOTrainingConfig(
        rollout_size=args.rollout_size,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )
    selector = PPODetectorSelector(
        detector_priors=priors,
        hidden_size=args.hidden_size,
    )

    print("Training RL selector with PPO...")
    result = train_ppo_selector(
        selector=selector,
        priors=priors,
        dataset_examples=examples,
        episodes=args.episodes,
        seed=args.seed,
        config=training_config,
    )
    summary = result["summary"]
    baseline_by_name = {
        name: contextual_static_baseline(priors, name, examples)
        for name in priors.keys()
    }
    best_contextual_name = max(
        baseline_by_name,
        key=lambda name: baseline_by_name[name]["expected_accuracy"],
    )
    cheapest_name = min(priors.values(), key=lambda prior: prior.cost).name
    llm_name = "llm_judge" if "llm_judge" in priors else best_contextual_name

    baselines = {
        "best_contextual_accuracy": baseline_by_name[best_contextual_name],
        "cheapest": baseline_by_name[cheapest_name],
        "always_llm": baseline_by_name[llm_name],
    }

    payload = {
        "algorithm": "ppo",
        "config": {
            "algorithm": "ppo",
            "episodes": args.episodes,
            "limit": args.limit,
            "seed": args.seed,
            "rollout_size": args.rollout_size,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "clip_epsilon": args.clip_epsilon,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
        },
        "summary": summary,
        "baselines": baselines,
        "feature_schema": {
            "names": selector.feature_names,
        },
        "policy_model": selector.export_policy(),
        "training_history": result["history"],
        "action_counts": result["action_counts"],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL selector for hallucination detectors")
    parser.add_argument("--algorithm", choices=["bandit", "ppo"], default="ppo", help="RL algorithm to train")
    parser.add_argument("--episodes", type=int, default=40, help="Training passes over dataset")
    parser.add_argument("--limit", type=int, default=200, help="Number of PHANTOM examples to use")
    parser.add_argument("--epsilon", type=float, default=0.12, help="Bandit exploration rate")
    parser.add_argument("--prior-strength", type=int, default=25, help="Bandit pseudo-count strength from priors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rollout-size", type=int, default=128, help="PPO rollout size")
    parser.add_argument("--update-epochs", type=int, default=8, help="PPO optimization epochs per rollout")
    parser.add_argument("--minibatch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--hidden-size", type=int, default=32, help="PPO hidden layer size")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--value-coef", type=float, default=0.5, help="PPO value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="PPO entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="PPO gradient clipping norm")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/processed/benchmark_results.json"),
        help="Benchmark result JSON used as detector priors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/rl_policy.json"),
        help="Where to save trained policy snapshot",
    )

    args = parser.parse_args()

    if not args.benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {args.benchmark_path}. Run benchmark_detectors.py first."
        )

    print("Loading benchmark priors...")
    priors = load_benchmark_priors(args.benchmark_path)
    print(f"Loaded priors for detectors: {', '.join(priors.keys())}")

    print("Loading PHANTOM dataset examples...")
    dataset = load_test_dataset(use_phantom=True, limit=args.limit)
    examples = dataset["examples"]
    print(f"Loaded {len(examples)} examples")

    if args.algorithm == "ppo":
        payload = build_ppo_payload(args, priors, examples)
    else:
        payload = build_bandit_payload(args, priors, examples)

    save_policy(args.output, payload)
    print_summary(payload["summary"], payload["baselines"])
    print(f"\nPolicy saved to: {args.output}")


if __name__ == "__main__":
    main()
