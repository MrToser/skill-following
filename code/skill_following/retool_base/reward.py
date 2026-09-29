# LOCKED: false
"""Non-negative Retool training reward."""

from __future__ import annotations

from typing import Any

try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as exc:  # pragma: no cover - exercised in training env
    raise ImportError("MathDapo reward is required for Retool skill-following training") from exc

from slime.utils.types import Sample

from .models import RetoolRewardConfig


# Data: protocol counters and accepted sample state. Algorithm: diagnose whether the complete Retool skill trajectory finished.
def skill_protocol_completed(
    *,
    skill_loaded: bool,
    post_skill_think_completed: bool,
    post_python_think_completed: bool,
    tool_call_count: int,
    interpreter_count: int,
    successful_interpreter_count: int,
    sample_completed: bool,
    accepted_answer: str | None,
) -> bool:
    return (
        skill_loaded
        and post_skill_think_completed
        and post_python_think_completed
        and tool_call_count > 0
        and interpreter_count > 0
        and successful_interpreter_count > 0
        and sample_completed
        and accepted_answer is not None
    )


# Data: sample status enum or string. Algorithm: normalize either representation to its string value.
def get_sample_status(sample: Sample) -> str:
    status = getattr(sample, "status", "")
    if hasattr(status, "value"):
        return str(status.value)
    return str(status)


# Data: sample status and serialized state. Algorithm: expose only the final answer accepted by a completed rollout.
def get_accepted_answer(sample: Sample, state: dict[str, Any]) -> str | None:
    if get_sample_status(sample) != Sample.Status.COMPLETED.value or not bool(state.get("done")):
        return None
    accepted_answer = state.get("accepted_answer")
    if not isinstance(accepted_answer, str) or not accepted_answer.strip():
        return None
    return accepted_answer.strip()


# Data: accepted boxed answer. Algorithm: build the sole text visible to the strict mathematical verifier.
def build_math_verification_text(accepted_answer: str | None) -> str:
    if accepted_answer is None:
        return ""
    return f"Answer: \\boxed{{{accepted_answer}}}"


# Data: accepted answer, protocol state, and version-owned weights. Algorithm: reward successful paths and cap think deductions.
def compute_skill_reward(sample: Sample, config: RetoolRewardConfig) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    state = metadata.get("skill_protocol_state") or {}
    accepted_answer = get_accepted_answer(sample, state)
    sample_completed = get_sample_status(sample) == Sample.Status.COMPLETED.value
    solution_str = build_math_verification_text(accepted_answer)
    ground_truth = sample.label if sample.label is not None else ""
    math_result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    if math_result.get("pred") is None:
        math_result["pred"] = ""

    answer_correct = accepted_answer is not None and float(math_result.get("score", 0.0)) > 0
    skill_loaded = bool(state.get("skill_loaded"))
    post_skill_think_completed = bool(state.get("post_skill_think_completed"))
    post_python_think_completed = bool(state.get("post_python_think_completed"))
    successful_interpreter_count = int(state.get("successful_interpreter_count") or 0)
    tool_succeeded = successful_interpreter_count > 0
    repeated_think_count = max(0, int(state.get("repeated_think_count") or 0))
    used_fields = list(state.get("used_fields") or [])
    formatted_answer_after_think = (
        accepted_answer is not None
        and len(used_fields) >= 2
        and used_fields[-2:] == ["think", "answer"]
    )
    protocol_completed = skill_protocol_completed(
        skill_loaded=skill_loaded,
        post_skill_think_completed=post_skill_think_completed,
        post_python_think_completed=post_python_think_completed,
        tool_call_count=int(state.get("tool_call_count") or 0),
        interpreter_count=int(state.get("interpreter_count") or 0),
        successful_interpreter_count=successful_interpreter_count,
        sample_completed=sample_completed,
        accepted_answer=accepted_answer,
    )

    repeated_think_penalty = min(
        repeated_think_count * config.repeated_think_penalty,
        config.max_repeated_think_penalty,
        config.stage_reward if post_skill_think_completed and tool_succeeded else 0.0,
    )
    reward_components = {
        "skill_call_reward": config.stage_reward if skill_loaded else 0.0,
        "post_skill_think_reward": (
            config.stage_reward - repeated_think_penalty
            if post_skill_think_completed and tool_succeeded
            else 0.0
        ),
        "python_reward": config.tool_success_reward if tool_succeeded else 0.0,
        "post_python_think_reward": (
            config.stage_reward if post_python_think_completed and tool_succeeded else 0.0
        ),
        "formatted_answer_reward": config.stage_reward if formatted_answer_after_think else 0.0,
        "answer_reward": config.answer_reward if answer_correct else 0.0,
    }
    total_reward = round(sum(reward_components.values()), 10)

    result = dict(math_result)
    result.update(
        {
            "score": total_reward,
            **reward_components,
            "answer_correct": answer_correct,
            "sample_completed": sample_completed,
            "answer_accepted": accepted_answer is not None,
            "accepted_answer": accepted_answer or "",
            "skill_loaded": skill_loaded,
            "think_count": int(state.get("think_count") or 0),
            "required_think_count": int(state.get("required_think_count") or 0),
            "completed_required_think_count": int(state.get("completed_required_think_count") or 0),
            "post_skill_think_completed": post_skill_think_completed,
            "post_python_think_completed": post_python_think_completed,
            "formatted_answer_after_think": formatted_answer_after_think,
            "tool_call_count": int(state.get("tool_call_count") or 0),
            "interpreter_count": int(state.get("interpreter_count") or 0),
            "successful_interpreter_count": successful_interpreter_count,
            "repeated_think_count": repeated_think_count,
            "repeated_think_penalty": repeated_think_penalty,
            "violation_count": len(state.get("violations") or []),
            "skill_protocol_completed": protocol_completed,
        }
    )
    return result
