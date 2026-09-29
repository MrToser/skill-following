# LOCKED: false
"""Deterministic regression checks for the public Retool base API."""

from __future__ import annotations

import argparse
import asyncio

from slime.utils.types import Sample

from .constants import SKILL_NAME
from .evaluation import compute_dual_eval_metrics
from .logger import LOGGER
from .models import ParsedAction, RetoolRewardConfig, RetoolSkillState
from .protocol import (
    execute_protocol_action,
    is_substantive_think,
    load_skill_sections,
    parse_action,
    parse_tool_call_json,
)
from .reward import build_math_verification_text, compute_skill_reward, get_accepted_answer
from .rollout import resolve_max_turns


REWARD_CONFIG = RetoolRewardConfig(
    stage_reward=0.20,
    tool_success_reward=0.50,
    answer_reward=1.00,
    repeated_think_penalty=0.05,
    max_repeated_think_penalty=0.20,
)


# Data: deterministic fixtures. Algorithm: validate parser, staged protocol state, reward, and eval aggregation without SGLang.
def self_test() -> None:
    sections = load_skill_sections()
    assert sections["description"].startswith(f"name: {SKILL_NAME}")
    assert parse_action(f"<skill_call>{SKILL_NAME}</skill_call>").field_name == "skill_call"
    assert parse_action("<think>I should inspect the arithmetic.</think>").field_name == "think"
    assert parse_action(f"Copy this: <skill_call>{SKILL_NAME}</skill_call>").field_name is None
    assert not is_substantive_think("...")
    assert is_substantive_think("I should inspect the arithmetic.")
    tool = '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(2+2)"}}</tool_call>'
    parsed = parse_action(tool)
    assert parsed.field_name == "tool_call"
    code, error = parse_tool_call_json(parsed.content)
    assert error is None and code == "print(2+2)"
    assert parse_action("Answer: \\boxed{4}").field_name == "answer"
    assert resolve_max_turns(argparse.Namespace(max_turns=None), 16) == 16
    gated_state = RetoolSkillState()
    action_result = asyncio.run(
        execute_protocol_action(ParsedAction("skill_call", SKILL_NAME), gated_state)
    )
    assert not action_result.done and "<skill_body>" in action_result.observation
    assert action_result.violation is None and not gated_state.violations
    assert gated_state.must_think and gated_state.required_think_count == 1
    assert gated_state.pending_think_source == "skill"
    action_result = asyncio.run(
        execute_protocol_action(ParsedAction("think", "..."), gated_state)
    )
    assert not action_result.done and action_result.violation is not None
    assert action_result.violation.code == "non_substantive_think"
    assert "non_substantive_think" in gated_state.violations
    action_result = asyncio.run(
        execute_protocol_action(
            ParsedAction("think", "I will compute the requested quantity exactly with Python."),
            gated_state,
        )
    )
    assert not action_result.done and action_result.violation is not None
    assert action_result.violation.code == "repeated_think" and not gated_state.must_think
    assert gated_state.post_skill_think_completed
    assert gated_state.repeated_think_count == 1
    action_result = asyncio.run(
        execute_protocol_action(ParsedAction("answer", "4"), gated_state)
    )
    assert not action_result.done and action_result.violation is not None
    assert action_result.violation.code == "answer_before_successful_interpreter"
    assert "answer_before_successful_interpreter" in gated_state.violations
    assert gated_state.accepted_answer is None

    repeated_think_state = RetoolSkillState(skill_loaded=True, used_fields=["skill_call"])
    action_result = asyncio.run(
        execute_protocol_action(
            ParsedAction("think", "I will compute the requested value with Python."),
            repeated_think_state,
        )
    )
    assert not action_result.done and action_result.violation is None
    assert not action_result.observation and repeated_think_state.repeated_think_count == 0
    action_result = asyncio.run(
        execute_protocol_action(
            ParsedAction("think", "I will now continue planning the same computation."),
            repeated_think_state,
        )
    )
    assert not action_result.done and action_result.violation is not None
    assert action_result.violation.code == "repeated_think"
    assert repeated_think_state.repeated_think_count == 1
    assert "repeated_think" in repeated_think_state.violations
    assert not repeated_think_state.must_think

    correction_state = RetoolSkillState(
        skill_loaded=True,
        used_fields=["skill_call", "think"],
        think_count=1,
    )
    action_result = asyncio.run(execute_protocol_action(ParsedAction(None, ""), correction_state))
    assert action_result.violation is not None and action_result.violation.code == "no_action"
    assert correction_state.used_fields[-1] == "no_action"
    correction_state.require_think("violation")
    action_result = asyncio.run(
        execute_protocol_action(
            ParsedAction("think", "I will correct the invalid action before using Python."),
            correction_state,
        )
    )
    assert action_result.violation is None
    assert correction_state.repeated_think_count == 0
    assert not correction_state.must_think

    completed_sample = argparse.Namespace(status=Sample.Status.COMPLETED)
    truncated_sample = argparse.Namespace(status=Sample.Status.TRUNCATED)
    accepted_state = {"done": True, "accepted_answer": "4"}
    rejected_state = {"done": False, "accepted_answer": None}
    assert get_accepted_answer(completed_sample, accepted_state) == "4"
    assert get_accepted_answer(truncated_sample, accepted_state) is None
    assert get_accepted_answer(completed_sample, rejected_state) is None
    assert build_math_verification_text("4") == "Answer: \\boxed{4}"
    assert build_math_verification_text(None) == ""

    full_protocol_state = {
        "skill_loaded": True,
        "used_fields": ["skill_call", "think", "tool_call", "think", "answer"],
        "post_skill_think_completed": True,
        "post_python_think_completed": True,
        "tool_call_count": 1,
        "interpreter_count": 1,
        "successful_interpreter_count": 1,
        "repeated_think_count": 0,
        "required_think_count": 2,
        "completed_required_think_count": 2,
        "done": True,
        "accepted_answer": "4",
        "violations": [],
    }
    full_protocol_sample = argparse.Namespace(
        status=Sample.Status.COMPLETED,
        metadata={"skill_protocol_state": full_protocol_state},
        label="4",
    )
    full_protocol_reward = compute_skill_reward(full_protocol_sample, REWARD_CONFIG)
    expected_full_score = (
        4 * REWARD_CONFIG.stage_reward
        + REWARD_CONFIG.tool_success_reward
        + REWARD_CONFIG.answer_reward
    )
    assert full_protocol_reward["score"] == expected_full_score
    assert full_protocol_reward["skill_call_reward"] == REWARD_CONFIG.stage_reward
    assert full_protocol_reward["post_skill_think_reward"] == REWARD_CONFIG.stage_reward
    assert full_protocol_reward["python_reward"] == REWARD_CONFIG.tool_success_reward
    assert full_protocol_reward["post_python_think_reward"] == REWARD_CONFIG.stage_reward
    assert full_protocol_reward["formatted_answer_reward"] == REWARD_CONFIG.stage_reward
    assert full_protocol_reward["answer_reward"] == REWARD_CONFIG.answer_reward
    assert full_protocol_reward["repeated_think_penalty"] == 0.0

    partial_protocol_state = dict(full_protocol_state)
    partial_protocol_state.update(
        {
            "post_python_think_completed": False,
            "tool_call_count": 0,
            "interpreter_count": 0,
            "successful_interpreter_count": 0,
            "done": False,
            "accepted_answer": None,
            "used_fields": ["skill_call", "think"],
        }
    )
    partial_protocol_sample = argparse.Namespace(
        status=Sample.Status.TRUNCATED,
        metadata={"skill_protocol_state": partial_protocol_state},
        label="4",
    )
    partial_protocol_reward = compute_skill_reward(partial_protocol_sample, REWARD_CONFIG)
    assert partial_protocol_reward["score"] == REWARD_CONFIG.stage_reward
    assert partial_protocol_reward["skill_call_reward"] == REWARD_CONFIG.stage_reward
    assert partial_protocol_reward["post_skill_think_reward"] == 0.0
    assert partial_protocol_reward["answer_reward"] == 0.0

    failed_tool_state = dict(full_protocol_state)
    failed_tool_state.update(
        {
            "used_fields": ["skill_call", "think", "tool_call", "think"],
            "successful_interpreter_count": 0,
            "done": False,
            "accepted_answer": None,
        }
    )
    failed_tool_sample = argparse.Namespace(
        status=Sample.Status.TRUNCATED,
        metadata={"skill_protocol_state": failed_tool_state},
        label="4",
    )
    failed_tool_reward = compute_skill_reward(failed_tool_sample, REWARD_CONFIG)
    assert failed_tool_reward["score"] == REWARD_CONFIG.stage_reward
    assert failed_tool_reward["python_reward"] == 0.0
    assert failed_tool_reward["post_python_think_reward"] == 0.0

    repeated_think_state = dict(full_protocol_state)
    repeated_think_state["repeated_think_count"] = 10
    repeated_think_sample = argparse.Namespace(
        status=Sample.Status.COMPLETED,
        metadata={"skill_protocol_state": repeated_think_state},
        label="4",
    )
    repeated_think_reward = compute_skill_reward(repeated_think_sample, REWARD_CONFIG)
    assert repeated_think_reward["repeated_think_penalty"] == REWARD_CONFIG.max_repeated_think_penalty
    assert abs(
        repeated_think_reward["score"]
        - (full_protocol_reward["score"] - REWARD_CONFIG.max_repeated_think_penalty)
    ) < 1e-12
    assert min(
        partial_protocol_reward["score"],
        failed_tool_reward["score"],
        repeated_think_reward["score"],
    ) >= 0.0

    direct_answer_sample = argparse.Namespace(
        status=Sample.Status.COMPLETED,
        metadata={"skill_protocol_state": {"done": True, "accepted_answer": "4"}},
        label="4",
    )
    direct_answer_reward = compute_skill_reward(direct_answer_sample, REWARD_CONFIG)
    assert direct_answer_reward["answer_correct"]
    assert direct_answer_reward["score"] == REWARD_CONFIG.answer_reward
    assert direct_answer_reward["formatted_answer_reward"] == 0.0

    format_only_sample = argparse.Namespace(
        status=Sample.Status.COMPLETED,
        metadata={
            "skill_protocol_state": {
                "used_fields": ["think", "answer"],
                "think_count": 1,
                "done": True,
                "accepted_answer": "5",
            }
        },
        label="4",
    )
    format_only_reward = compute_skill_reward(format_only_sample, REWARD_CONFIG)
    assert not format_only_reward["skill_protocol_completed"]
    assert format_only_reward["score"] == REWARD_CONFIG.stage_reward
    assert format_only_reward["formatted_answer_reward"] == REWARD_CONFIG.stage_reward
    assert format_only_reward["answer_reward"] == 0.0

    wrong_answer_state = dict(full_protocol_state)
    wrong_answer_state["accepted_answer"] = "5"
    wrong_answer_sample = argparse.Namespace(
        status=Sample.Status.COMPLETED,
        metadata={"skill_protocol_state": wrong_answer_state},
        label="4",
    )
    wrong_answer_reward = compute_skill_reward(wrong_answer_sample, REWARD_CONFIG)
    assert wrong_answer_reward["score"] == (
        4 * REWARD_CONFIG.stage_reward + REWARD_CONFIG.tool_success_reward
    )
    assert wrong_answer_reward["formatted_answer_reward"] == REWARD_CONFIG.stage_reward
    assert wrong_answer_reward["answer_reward"] == 0.0
    eval_samples = [
        argparse.Namespace(
            reward={
                "score": 2.3,
                "answer_reward": 1.0,
                "skill_loaded": True,
                "post_skill_think_completed": True,
                "post_python_think_completed": True,
                "formatted_answer_after_think": True,
                "skill_call_reward": 0.2,
                "post_skill_think_reward": 0.2,
                "python_reward": 0.5,
                "post_python_think_reward": 0.2,
                "formatted_answer_reward": 0.2,
                "answer_accepted": True,
                "tool_call_count": 1,
                "successful_interpreter_count": 1,
                "repeated_think_count": 0,
                "repeated_think_penalty": 0.0,
                "skill_protocol_completed": True,
            }
        ),
        argparse.Namespace(
            reward={
                "score": 1.1,
                "answer_reward": 0.0,
                "skill_loaded": True,
                "post_skill_think_completed": True,
                "post_python_think_completed": True,
                "formatted_answer_after_think": True,
                "skill_call_reward": 0.2,
                "post_skill_think_reward": 0.0,
                "python_reward": 0.5,
                "post_python_think_reward": 0.2,
                "formatted_answer_reward": 0.2,
                "answer_accepted": True,
                "tool_call_count": 1,
                "successful_interpreter_count": 1,
                "repeated_think_count": 5,
                "repeated_think_penalty": 0.2,
                "skill_protocol_completed": True,
            }
        ),
    ]
    eval_metrics, math_correctness = compute_dual_eval_metrics(eval_samples)
    assert abs(eval_metrics["protocol_reward"] - 1.7) < 1e-12
    assert eval_metrics["math_accuracy"] == 0.5
    assert eval_metrics["skill_call_stage_rate"] == 1.0
    assert eval_metrics["post_skill_think_stage_rate"] == 1.0
    assert eval_metrics["post_python_think_stage_rate"] == 1.0
    assert eval_metrics["formatted_answer_stage_rate"] == 1.0
    assert eval_metrics["repeated_think_rate"] == 0.5
    assert eval_metrics["repeated_think_penalty"] == 0.1
    assert eval_metrics["protocol_completion_rate"] == 1.0
    assert math_correctness == [1.0, 0.0]
    LOGGER.info("self-test passed")
