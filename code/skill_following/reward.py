# LOCKED: false
"""两任务同尺度 transition reward 与任务 verifier。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qa_em_format import em_check
from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from slime.utils.types import Sample

from .config import get_runtime_config
from .contracts import (
    ACTION_TRANSITION,
    ANSWER_TRANSITION,
    CORE_TRANSITION_NAMES,
    EVIDENCE_THINK_TRANSITION,
    OBSERVATION_TRANSITION,
    PLAN_TRANSITION,
    RESEARCH_TRANSITION,
    SKILL_TRANSITION,
    SUMMARY_TRANSITION,
    TRANSITION_NAMES,
    TaskType,
    parse_training_label,
)
from .logger import my_logger


LOGGER = my_logger("skill_following_experiment_suite.reward")
MATH_TRANSITION_SPECS = (
    (SKILL_TRANSITION, "skill_transition_reward", 0.10),
    (PLAN_TRANSITION, "plan_transition_reward", 0.10),
    (ACTION_TRANSITION, "action_transition_reward", 0.20),
    (OBSERVATION_TRANSITION, "observation_transition_reward", 0.20),
    (EVIDENCE_THINK_TRANSITION, "evidence_think_transition_reward", 0.20),
    (ANSWER_TRANSITION, "answer_transition_reward", 0.20),
)
SEARCH_TRANSITION_SPECS = (
    (SKILL_TRANSITION, "skill_transition_reward", 0.10),
    (PLAN_TRANSITION, "plan_transition_reward", 0.10),
    (ACTION_TRANSITION, "action_transition_reward", 0.20),
    (OBSERVATION_TRANSITION, "observation_transition_reward", 0.20),
    (EVIDENCE_THINK_TRANSITION, "evidence_think_transition_reward", 0.10),
    (SUMMARY_TRANSITION, "summary_transition_reward", 0.10),
    (ANSWER_TRANSITION, "answer_transition_reward", 0.20),
)
TRANSITION_REWARD_FIELDS = {
    transition_name: reward_field
    for transition_name, reward_field, _ in (*MATH_TRANSITION_SPECS, *SEARCH_TRANSITION_SPECS)
}
TRANSITION_REWARD_FIELDS[RESEARCH_TRANSITION] = "research_transition_reward"


# 数据：样本任务类型。算法：Math 使用六阶段权重，Search 把证据思考的 0.20 拆给可选 summary 0.10。
def transition_specs_for_task(
    task_type: str, first_research_reward: float = 0.0
) -> tuple[tuple[str, str, float], ...]:
    if task_type == TaskType.MATH.value:
        return MATH_TRANSITION_SPECS
    assert task_type == TaskType.SEARCH.value
    return (*SEARCH_TRANSITION_SPECS, (RESEARCH_TRANSITION, "research_transition_reward", first_research_reward))


# 数据：Sample status enum 或字符串。算法：统一成字符串值供 reward 判断。
def sample_status_value(sample: Sample) -> str:
    status = getattr(sample, "status", "")
    return str(status.value) if hasattr(status, "value") else str(status)


# 数据：完成状态与序列化 Harness state。算法：只暴露真正被状态机接受的终止答案。
def accepted_answer(sample: Sample, state: dict[str, Any]) -> str | None:
    if sample_status_value(sample) != Sample.Status.COMPLETED.value or not bool(state.get("done")):
        return None
    answer = state.get("accepted_answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    return answer.strip()


# 数据：统一 label、任务类型和已接受答案。算法：按任务分派 MathDapo 或 QA EM verifier。
def compute_outcome(sample: Sample, answer: str | None) -> tuple[bool, dict[str, Any]]:
    label = parse_training_label(sample.label)
    if answer is None:
        return False, {"prediction": "", "targets": list(label.targets)}
    if label.task_type == TaskType.MATH.value:
        math_result = math_dapo_compute_score(
            f"Answer: \\boxed{{{answer}}}",
            label.targets[0],
            strict_box_verify=True,
        )
        correct = float(math_result.get("score", 0.0)) > 0
        return correct, {
            "prediction": str(math_result.get("pred") or ""),
            "targets": list(label.targets),
        }
    correct = bool(em_check(answer, list(label.targets)))
    return correct, {"prediction": answer, "targets": list(label.targets)}


# 数据：序列化 transition recovery map。算法：拒绝未知 transition 和非布尔 recovery 标记。
def validated_transition_recovery(state: dict[str, Any]) -> dict[str, bool]:
    transition_recovery = state.get("transition_recovery") or {}
    if not isinstance(transition_recovery, dict):
        raise TypeError("transition_recovery must be a dict")
    unknown = set(transition_recovery) - set(TRANSITION_NAMES)
    if unknown:
        raise ValueError(f"unknown transitions: {sorted(unknown)}")
    if not all(isinstance(value, bool) for value in transition_recovery.values()):
        raise TypeError("transition_recovery values must be booleans")
    return dict(transition_recovery)


# 数据：一个首次 transition、recovery 上下文和权重。算法：仅对该 transition 局部乘0.9。
def transition_reward(
    transition_recovery: dict[str, bool],
    transition_name: str,
    weight: float,
    recovery_factor: float,
) -> float:
    if transition_name not in transition_recovery:
        return 0.0
    factor = recovery_factor if transition_recovery[transition_name] else 1.0
    return round(weight * factor, 10)


# 数据：一条 rollout Sample。算法：按任务累加首次 transition，并独立加入任务正确性 1 分。
def compute_reward(sample: Sample) -> dict[str, Any]:
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a slime Sample")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    state = metadata.get("skill_protocol_state") or {}
    if not isinstance(state, dict):
        raise TypeError("skill_protocol_state must be a dict")

    runtime = get_runtime_config()
    if state:
        expected_runtime_fields = {
            "experiment_id": runtime.experiment_id,
            "skill_variant": runtime.skill_variant,
            "state_aware_hints": runtime.state_aware_hints,
            "transition_credit_enabled": runtime.transition_credit,
            "recovery_factor": runtime.recovery_factor,
            "first_research_reward": runtime.first_research_reward,
            "observation_mode": runtime.observation_mode,
        }
        for field_name, expected_value in expected_runtime_fields.items():
            if state.get(field_name) != expected_value:
                raise ValueError(
                    f"state {field_name}={state.get(field_name)!r} does not match "
                    f"runtime {expected_value!r}"
                )
    label = parse_training_label(sample.label)
    state_task_type = str(state.get("expected_task_type") or label.task_type)
    if state and state_task_type != label.task_type:
        raise ValueError("state task_type must match label task_type")

    answer = accepted_answer(sample, state)
    outcome_correct, verifier = compute_outcome(sample, answer)
    transition_recovery = validated_transition_recovery(state)
    component_rewards = {
        reward_field: 0.0 for reward_field in TRANSITION_REWARD_FIELDS.values()
    }
    transition_completed = {
        f"{transition_name}_completed": transition_name in transition_recovery
        for transition_name in TRANSITION_NAMES
    }
    raw_protocol_progress = 0.0
    shaped_protocol_progress = 0.0
    for transition_name, reward_field, weight in transition_specs_for_task(
        label.task_type, runtime.first_research_reward
    ):
        eligible_reward = transition_reward(
            transition_recovery,
            transition_name,
            weight,
            runtime.recovery_factor,
        )
        component_rewards[reward_field] = eligible_reward if runtime.transition_credit else 0.0
        if transition_name in transition_recovery:
            raw_protocol_progress += weight
            shaped_protocol_progress += eligible_reward

    protocol_progress_reward = round(sum(component_rewards.values()), 10)
    raw_protocol_progress_reward = round(raw_protocol_progress, 10)
    eligible_protocol_progress_reward = round(shaped_protocol_progress, 10)
    outcome_reward = 1.0 if outcome_correct else 0.0
    protocol_completed = (
        all(name in transition_recovery for name in CORE_TRANSITION_NAMES)
        and sample_status_value(sample) == Sample.Status.COMPLETED.value
        and answer is not None
    )
    evidence_grounded = (
        bool(state.get("valid_observation_count"))
        and bool(state.get("evidence_think_count"))
        and answer is not None
        and not bool(state.get("observation_withheld_count"))
    )
    result = {
        "score": round(protocol_progress_reward + outcome_reward, 10),
        "task_type": label.task_type,
        "outcome_reward": outcome_reward,
        "outcome_correct": outcome_correct,
        "protocol_progress_reward": protocol_progress_reward,
        "raw_protocol_progress_reward": raw_protocol_progress_reward,
        "eligible_protocol_progress_reward": eligible_protocol_progress_reward,
        "recovery_discount": round(
            raw_protocol_progress_reward - eligible_protocol_progress_reward,
            10,
        ),
        "withheld_transition_credit": round(
            eligible_protocol_progress_reward - protocol_progress_reward,
            10,
        ),
        "transition_credit_enabled": runtime.transition_credit,
        "state_aware_hints": runtime.state_aware_hints,
        "recovery_factor": runtime.recovery_factor,
        "first_research_reward": runtime.first_research_reward,
        "skill_variant": runtime.skill_variant,
        "observation_mode": runtime.observation_mode,
        **component_rewards,
        **transition_completed,
        "recorded_transition_count": len(transition_recovery),
        "recovered_transition_count": sum(transition_recovery.values()),
        "used_recovery_hint": bool(state.get("used_recovery_hint")),
        "recovery_hint_count": max(0, int(state.get("recovery_hint_count") or 0)),
        "protocol_completed": protocol_completed,
        "evidence_grounded": evidence_grounded,
        "answer_accepted": answer is not None,
        "accepted_answer": answer or "",
        "prediction": verifier["prediction"],
        "targets": verifier["targets"],
        "action_count": max(0, int(state.get("action_count") or 0)),
        "valid_observation_count": max(0, int(state.get("valid_observation_count") or 0)),
        "invalid_observation_count": max(0, int(state.get("invalid_observation_count") or 0)),
        "observation_withheld_count": max(
            0,
            int(state.get("observation_withheld_count") or 0),
        ),
        "summary_count": max(0, int(state.get("summary_count") or 0)),
        "wrong_skill_count": max(0, int(state.get("wrong_skill_count") or 0)),
        "forged_observation_count": max(0, int(state.get("forged_observation_count") or 0)),
        "direct_answer_echo_count": max(0, int(state.get("direct_answer_echo_count") or 0)),
        "action_tail_trimmed_count": max(0, int(state.get("action_tail_trimmed_count") or 0)),
        "discarded_action_tail_token_count": max(
            0,
            int(state.get("discarded_action_tail_token_count") or 0),
        ),
        "feedback_message_count": max(0, int(state.get("feedback_message_count") or 0)),
        "feedback_character_count": max(0, int(state.get("feedback_character_count") or 0)),
        "violation_count": len(state.get("violations") or []),
        "first_accepted_action_turn": state.get("first_accepted_action_turn"),
        "injected_skill_source": state.get("injected_skill_source"),
        "injected_skill_body_sha256": state.get("injected_skill_body_sha256"),
    }
    return result


# 数据：slime 单 Sample reward callback。算法：返回含 score 的统一诊断字典。
async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    del args, kwargs
    return compute_reward(sample)


# 数据：同一 eval dataset 的 samples。算法：聚合协议、grounding、正确率与多工具行为。
def compute_eval_metrics(samples: list[Sample]) -> tuple[dict[str, float], list[float]]:
    assert samples
    rewards: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample.reward, dict):
            raise TypeError("unified eval requires reward dicts")
        rewards.append(sample.reward)
    sample_count = len(rewards)
    correctness = [float(bool(reward["outcome_correct"])) for reward in rewards]
    metrics = {
        "reward": sum(float(reward["score"]) for reward in rewards) / sample_count,
        "protocol_reward": sum(float(reward["protocol_progress_reward"]) for reward in rewards)
        / sample_count,
        "outcome_accuracy": sum(correctness) / sample_count,
        "protocol_completion_rate": sum(bool(reward["protocol_completed"]) for reward in rewards)
        / sample_count,
        "recovery_free_protocol_completion_rate": sum(
            bool(reward["protocol_completed"]) and not bool(reward["used_recovery_hint"])
            for reward in rewards
        ) / sample_count,
        "evidence_grounding_rate": sum(bool(reward["evidence_grounded"]) for reward in rewards)
        / sample_count,
        "clean_start_rate": sum(not bool(reward["used_recovery_hint"]) for reward in rewards)
        / sample_count,
        "skill_stage_rate": sum(bool(reward["skill_completed"]) for reward in rewards)
        / sample_count,
        "action_stage_rate": sum(bool(reward["action_completed"]) for reward in rewards)
        / sample_count,
        "observation_stage_rate": sum(bool(reward["observation_completed"]) for reward in rewards)
        / sample_count,
        "summary_stage_rate": sum(bool(reward["summary_completed"]) for reward in rewards)
        / sample_count,
        "research_stage_rate": sum(bool(reward["research_completed"]) for reward in rewards)
        / sample_count,
        "summary_reward": sum(float(reward["summary_transition_reward"]) for reward in rewards)
        / sample_count,
        "research_reward": sum(float(reward["research_transition_reward"]) for reward in rewards)
        / sample_count,
        "multi_action_rate": sum(int(reward["action_count"]) > 1 for reward in rewards) / sample_count,
        "invalid_observation_rate": sum(int(reward["invalid_observation_count"]) > 0 for reward in rewards)
        / sample_count,
        "wrong_skill_rate": sum(int(reward["wrong_skill_count"]) > 0 for reward in rewards) / sample_count,
        "hint_rate": sum(bool(reward["used_recovery_hint"]) for reward in rewards) / sample_count,
        "mean_feedback_messages": sum(int(reward["feedback_message_count"]) for reward in rewards)
        / sample_count,
        "mean_feedback_characters": sum(int(reward["feedback_character_count"]) for reward in rewards)
        / sample_count,
        "mean_invalid_transitions": sum(int(reward["violation_count"]) for reward in rewards)
        / sample_count,
        "withheld_observation_rate": sum(
            int(reward["observation_withheld_count"]) > 0 for reward in rewards
        )
        / sample_count,
    }
    return metrics, correctness


# 数据：NumPy、枚举和嵌套容器。算法：递归转换为可稳定写入 JSON 的值。
def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return json_safe(value.value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


# 数据：评测路径与 JSON 对象。算法：原子替换摘要，避免中断后留下半个 JSON。
def write_json(path: str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


# 数据：各数据集 Sample 与指标。算法：保存逐样本完整轨迹和可重算汇总。
def persist_eval_artifacts(
    rollout_id: int,
    data: dict[str, dict[str, Any]],
    metrics_by_dataset: dict[str, dict[str, float]],
) -> None:
    trace_path = os.environ.get("SF_EVAL_TRACE_PATH", "").strip()
    summary_path = os.environ.get("SF_EVAL_SUMMARY_PATH", "").strip()
    runtime = get_runtime_config()
    if trace_path:
        records: list[dict[str, Any]] = []
        for dataset_name, dataset_data in data.items():
            for sample in dataset_data.get("samples") or []:
                records.append(
                    {
                        "experiment_id": runtime.experiment_id,
                        "seed": runtime.seed,
                        "rollout_id": rollout_id,
                        "dataset": dataset_name,
                        "sample_index": getattr(sample, "index", None),
                        "group_index": getattr(sample, "group_index", None),
                        "status": sample_status_value(sample),
                        "prompt": getattr(sample, "prompt", None),
                        "response": getattr(sample, "response", ""),
                        "label": getattr(sample, "label", None),
                        "metadata": getattr(sample, "metadata", None),
                        "reward": getattr(sample, "reward", None),
                    }
                )
        destination = Path(trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    if summary_path:
        write_json(
            summary_path,
            {
                "experiment_id": runtime.experiment_id,
                "seed": runtime.seed,
                "rollout_id": rollout_id,
                "runtime": {
                    "task_mode": runtime.task_mode,
                    "skill_variant": runtime.skill_variant,
                    "state_aware_hints": runtime.state_aware_hints,
                    "transition_credit": runtime.transition_credit,
                    "recovery_factor": runtime.recovery_factor,
                    "observation_mode": runtime.observation_mode,
                },
                "datasets": metrics_by_dataset,
            },
        )


# 数据：slime eval rollout payload。算法：按 dataset 保留同尺度指标和通用 pass@k。
def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, dict_add_prefix

    log_dict = dict(extra_metrics or {})
    metrics_by_dataset: dict[str, dict[str, float]] = {}
    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"eval dataset {dataset_name!r} has no samples")
        metrics, correctness = compute_eval_metrics(samples)
        metrics_by_dataset[dataset_name] = metrics
        prefix = f"eval/{dataset_name}/"
        log_dict |= dict_add_prefix(metrics, prefix)
        log_dict[f"eval/{dataset_name}"] = metrics["reward"]
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), prefix)

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
            log_dict[f"{prefix}pass@1"] = metrics["outcome_accuracy"]
        elif len(correctness) % group_size == 0:
            log_dict |= dict_add_prefix(compute_pass_rate(correctness, group_size), prefix)

    persist_eval_artifacts(rollout_id, data, metrics_by_dataset)
    LOGGER.info("eval %s: %s", rollout_id, log_dict)
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
