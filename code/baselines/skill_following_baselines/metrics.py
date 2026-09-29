# LOCKED: false
"""Outcome-only verifier and artifact logger shared by all baseline evaluations."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from qa_em_format import em_check
from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from slime.utils.types import Sample


ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_LINE = re.compile(r"(?:final\s+)?answer\s*:\s*(.+)", re.IGNORECASE)


def _targets_and_task(sample: Sample) -> tuple[str, list[str]]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw = sample.label
    if isinstance(raw, str):
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
    else:
        payload = raw
    task_type = str(metadata.get("task_type") or "").strip().lower()
    if isinstance(payload, dict) and "ground_truth" in payload:
        ground_truth = payload.get("ground_truth") or {}
        targets = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
        task_type = task_type or "search"
    elif isinstance(payload, dict):
        targets = payload.get("target")
        task_type = task_type or str(payload.get("task_type") or "").strip().lower()
    else:
        targets = payload
        task_type = task_type or "math"
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    if targets is None:
        target_items: list[Any] = []
    elif isinstance(targets, (list, tuple, set)):
        target_items = list(targets)
    else:
        target_items = [targets]
    normalized = [str(item).strip() for item in target_items if str(item).strip()]
    assert task_type in {"math", "search"} and normalized
    return task_type, normalized


def extract_search_answer(response: str) -> str:
    tagged = ANSWER_TAG.findall(response)
    if tagged:
        return tagged[-1].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    for line in reversed(lines):
        match = ANSWER_LINE.search(line)
        if match:
            return match.group(1).strip().strip("*`")
    return lines[-1].strip("*`") if lines else ""


# 数据：一条 baseline Sample。算法：只计算任务正确性，不授予协议或工具奖励。
def compute_reward(sample: Sample) -> dict[str, Any]:
    assert isinstance(sample, Sample)
    task_type, targets = _targets_and_task(sample)
    response = str(sample.response or "")
    if task_type == "math":
        result = math_dapo_compute_score(response, targets[0], strict_box_verify=True)
        correct = float(result.get("score", 0.0)) > 0
        prediction = str(result.get("pred") or "")
    else:
        prediction = extract_search_answer(response)
        correct = bool(em_check(prediction, targets))
    return {
        "score": 1.0 if correct else 0.0,
        "outcome_correct": correct,
        "prediction": prediction,
        "targets": targets,
        "task_type": task_type,
        "baseline_condition": os.environ.get("BASELINE_CONDITION", "unknown"),
    }


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    del args, kwargs
    return compute_reward(sample)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


# 数据：slime eval payload。算法：统一保存 pass@1、完整轨迹与逐数据集样本数。
def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

    log_dict = dict(extra_metrics or {})
    summaries: dict[str, dict[str, float]] = {}
    trace_records: list[dict[str, Any]] = []
    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples") or []
        assert samples
        correctness = [float(bool(sample.reward["outcome_correct"])) for sample in samples]
        accuracy = sum(correctness) / len(correctness)
        summaries[dataset_name] = {"sample_count": len(samples), "outcome_accuracy": accuracy}
        prefix = f"eval/{dataset_name}/"
        log_dict[f"{prefix}outcome_accuracy"] = accuracy
        log_dict[f"{prefix}pass@1"] = accuracy
        log_dict[f"eval/{dataset_name}"] = accuracy
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), prefix)
        for sample in samples:
            trace_records.append(
                {
                    "condition": os.environ.get("BASELINE_CONDITION", "unknown"),
                    "dataset": dataset_name,
                    "index": getattr(sample, "index", None),
                    "prompt": sample.prompt,
                    "response": sample.response,
                    "label": sample.label,
                    "metadata": sample.metadata,
                    "reward": sample.reward,
                    "status": getattr(sample.status, "value", str(sample.status)),
                }
            )
    trace_path = os.environ.get("BASELINE_EVAL_TRACE_PATH", "").strip()
    if trace_path:
        destination = Path(trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as output:
            for record in trace_records:
                output.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
    summary_path = os.environ.get("BASELINE_EVAL_SUMMARY_PATH", "").strip()
    if summary_path:
        _write_json(
            summary_path,
            {
                "condition": os.environ.get("BASELINE_CONDITION", "unknown"),
                "rollout_id": rollout_id,
                "datasets": summaries,
            },
        )
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True


__all__ = ["compute_reward", "extract_search_answer", "log_eval_rollout_data", "reward_func"]
