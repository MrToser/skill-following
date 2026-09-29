# LOCKED: false
"""Retool evaluation aggregation and logging."""

from __future__ import annotations

from typing import Any, Callable

from slime.utils.types import Sample

from .logger import LOGGER


# Data: eval samples with reward dicts. Algorithm: aggregate protocol score and paper-style math accuracy separately.
def compute_dual_eval_metrics(samples: list[Sample]) -> tuple[dict[str, float], list[float]]:
    assert samples, "eval samples must not be empty"
    required_keys = {
        "score",
        "answer_reward",
        "skill_loaded",
        "post_skill_think_completed",
        "post_python_think_completed",
        "formatted_answer_after_think",
        "answer_accepted",
        "tool_call_count",
        "successful_interpreter_count",
        "repeated_think_count",
        "repeated_think_penalty",
        "skill_protocol_completed",
    }
    rewards: list[dict[str, Any]] = []
    for sample in samples:
        reward = sample.reward
        if not isinstance(reward, dict):
            raise TypeError("Retool dual eval requires each sample.reward to be a dict.")
        missing_keys = required_keys - reward.keys()
        if missing_keys:
            raise KeyError(f"Retool dual eval reward is missing keys: {sorted(missing_keys)}")
        rewards.append(reward)

    sample_count = len(rewards)
    math_correctness = [float(float(reward["answer_reward"]) > 0) for reward in rewards]
    metrics = {
        "protocol_reward": sum(float(reward["score"]) for reward in rewards) / sample_count,
        "math_accuracy": sum(math_correctness) / sample_count,
        "skill_call_stage_rate": sum(bool(reward["skill_loaded"]) for reward in rewards) / sample_count,
        "post_skill_think_stage_rate": sum(bool(reward["post_skill_think_completed"]) for reward in rewards)
        / sample_count,
        "post_python_think_stage_rate": sum(bool(reward["post_python_think_completed"]) for reward in rewards)
        / sample_count,
        "formatted_answer_stage_rate": sum(bool(reward["formatted_answer_after_think"]) for reward in rewards)
        / sample_count,
        "answer_accepted_rate": sum(bool(reward["answer_accepted"]) for reward in rewards) / sample_count,
        "tool_call_rate": sum(int(reward["tool_call_count"]) > 0 for reward in rewards) / sample_count,
        "interpreter_success_rate": sum(int(reward["successful_interpreter_count"]) > 0 for reward in rewards)
        / sample_count,
        "repeated_think_rate": sum(int(reward["repeated_think_count"]) > 0 for reward in rewards) / sample_count,
        "repeated_think_penalty": sum(float(reward["repeated_think_penalty"]) for reward in rewards)
        / sample_count,
        "protocol_completion_rate": sum(bool(reward["skill_protocol_completed"]) for reward in rewards)
        / sample_count,
    }
    return metrics, math_correctness


# Data: eval rollout payload and an explicit version-owned aggregator.
# Algorithm: log selected diagnostics and standard math accuracy/pass@k on identical samples.
def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
    metrics_fn: Callable[[list[Sample]], tuple[dict[str, float], list[float]]] = compute_dual_eval_metrics,
) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, dict_add_prefix

    log_dict = dict(extra_metrics or {})
    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Eval dataset {dataset_name!r} must expose non-empty samples.")
        dual_metrics, math_correctness = metrics_fn(samples)
        metric_prefix = f"eval/{dataset_name}/"
        log_dict |= dict_add_prefix(dual_metrics, metric_prefix)
        log_dict[f"eval/{dataset_name}"] = dual_metrics["protocol_reward"]
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), metric_prefix)

        truncated = dataset_data.get("truncated")
        if truncated is not None:
            log_dict[f"eval/{dataset_name}-truncated_ratio"] = sum(truncated) / len(truncated)

        dataset_config = next(
            (
                config
                for config in (getattr(args, "eval_datasets", None) or [])
                if getattr(config, "name", None) == dataset_name
            ),
            None,
        )
        group_size = int(
            getattr(dataset_config, "n_samples_per_eval_prompt", None)
            or getattr(args, "n_samples_per_eval_prompt", 1)
            or 1
        )
        if group_size == 1:
            log_dict[f"{metric_prefix}math_pass@1"] = dual_metrics["math_accuracy"]
        elif len(math_correctness) % group_size == 0:
            pass_metrics = compute_pass_rate(math_correctness, group_size)
            log_dict |= {
                f"{metric_prefix}math_{metric_name}": metric_value
                for metric_name, metric_value in pass_metrics.items()
            }

    LOGGER.info("eval %s: %s", rollout_id, log_dict)
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
