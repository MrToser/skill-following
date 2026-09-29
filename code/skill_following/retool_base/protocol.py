# LOCKED: false
"""Retool action grammar, state transitions, and tool execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

from .constants import (
    ACTION_PATTERN,
    ANSWER_PATTERN,
    DEFAULT_SKILL_PATH,
    MIN_SUBSTANTIVE_THINK_CHARS,
    SECTION_PATTERN,
    SKILL_NAME,
    STOP_TAGS,
)
from .models import ActionResult, ParsedAction, RetoolSkillState, ViolationEvent


_SKILL_BODY_CACHE: str | None = None


# Data: skill markdown path. Algorithm: extract marker-delimited sections and validate Retool invariants.
def load_skill_sections(skill_path: Path = DEFAULT_SKILL_PATH) -> dict[str, str]:
    assert skill_path.exists(), f"skill file does not exist: {skill_path}"
    text = skill_path.read_text(encoding="utf-8")
    sections = {name: body.strip() for name, body in SECTION_PATTERN.findall(text)}
    assert set(sections) == {"description", "skill_body"}, f"invalid skill sections: {sorted(sections)}"
    assert sections["description"].startswith(f"name: {SKILL_NAME}"), "description must expose skill name"
    assert "Execute code-interpreter-protocol" in sections["skill_body"], "skill body must define execution"
    assert "problem-specific reasoning" in sections["skill_body"], "skill body must require substantive reasoning"
    assert "<tool_call>" in sections["skill_body"], "skill body must define Retool tool_call"
    assert "<interpreter>" in sections["skill_body"], "skill body must define interpreter observation"
    assert "at least one successful interpreter call" in sections["skill_body"], "skill body must require tool evidence"
    assert "Answer: \\boxed" in sections["skill_body"], "skill body must define final answer format"
    return sections


# Data: skill_body section text. Algorithm: cache pure body text and keep outer tag owned by harness.
def load_skill_body_text() -> str:
    global _SKILL_BODY_CACHE
    if _SKILL_BODY_CACHE is None:
        body = load_skill_sections()["skill_body"]
        assert "<skill_body>" not in body and "</skill_body>" not in body, "skill body must not include outer tag"
        _SKILL_BODY_CACHE = body
    return _SKILL_BODY_CACHE


# Data: pure skill body. Algorithm: wrap as environment observation with loss_mask=0.
def build_skill_body_observation() -> str:
    return f"\n\n<skill_body>{load_skill_body_text()}</skill_body>\n\n"


# Data: current state, violation code, and immutable context. Algorithm: record diagnostics and return a text-free event.
def record_violation(
    state: RetoolSkillState,
    violation: str,
    **context: str,
) -> ActionResult:
    assert violation.strip(), "violation must not be empty"
    state.violations.append(violation)
    event = ViolationEvent(code=violation, context=tuple(sorted(context.items())))
    return ActionResult(violation=event)


# Data: sampling params. Algorithm: merge protocol stop tags without dropping caller-provided stops.
def with_protocol_stop_tags(sampling_params: dict[str, Any]) -> dict[str, Any]:
    existing_stop = sampling_params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    merged_stop = list(dict.fromkeys([*existing_stop, *STOP_TAGS]))
    return {**sampling_params, "stop": merged_stop}


# Data: raw model response. Algorithm: trim to the earliest known action boundary when text mode is used.
def postprocess_responses(response: str) -> str:
    assert isinstance(response, str), "response must be a string"
    end_positions = [response.find(tag) + len(tag) for tag in STOP_TAGS if tag in response]
    if end_positions:
        return response[: min(end_positions)]
    answer_match = ANSWER_PATTERN.search(response)
    if answer_match:
        return response[: answer_match.end()]
    return response


# Data: thought content. Algorithm: reject empty, punctuation-only, and short placeholder reasoning.
def is_substantive_think(content: str) -> bool:
    assert isinstance(content, str), "thought content must be a string"
    normalized = " ".join(content.strip().split())
    if len(normalized) < MIN_SUBSTANTIVE_THINK_CHARS:
        return False
    semantic_chars = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    return len(semantic_chars) >= 6


# Data: model output. Algorithm: classify only an action that begins the current model turn.
def parse_action(prediction: str) -> ParsedAction:
    assert isinstance(prediction, str), "prediction must be a string"
    stripped_prediction = prediction.strip()
    action_match = ACTION_PATTERN.match(stripped_prediction)
    answer_match = ANSWER_PATTERN.match(stripped_prediction)
    if action_match is None and answer_match is None:
        return ParsedAction(field_name=None, content="")
    if answer_match is not None and (action_match is None or answer_match.start() < action_match.start()):
        return ParsedAction(field_name="answer", content=answer_match.group(1).strip())
    assert action_match is not None
    return ParsedAction(field_name=action_match.group(1), content=action_match.group(2).strip())


# Data: text inside <tool_call>. Algorithm: parse Retool JSON schema and return executable Python code.
def parse_tool_call_json(content: str) -> tuple[str | None, str | None]:
    assert isinstance(content, str), "tool call content must be a string"
    try:
        payload = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "tool call payload must be a JSON object"
    if payload.get("name") != "code_interpreter":
        return None, "tool name must be code_interpreter"
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None, "arguments must be a JSON object"
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        return None, "arguments.code must be a non-empty string"
    return code.strip(), None


# Data: Python code. Algorithm: execute through the Retool sandbox and report whether execution succeeded.
async def execute_code_interpreter(code: str) -> tuple[str, bool]:
    normalized_code = code.strip()
    if not normalized_code:
        return "\n\n<interpreter>\nError: empty Python code.\n</interpreter>\n\n", False
    async with SEMAPHORE:
        result = await tool_registry.execute_tool("code_interpreter", {"code": normalized_code})
    succeeded = not str(result).lstrip().startswith("Error:")
    return f"\n\n<interpreter>\n{result}\n</interpreter>\n\n", succeeded


# Data: executable code and protocol state. Algorithm: execute once and update invocation/success counters together.
async def execute_and_record(code: str, state: RetoolSkillState) -> str:
    state.tool_call_count += 1
    observation, succeeded = await execute_code_interpreter(code)
    state.interpreter_count += 1
    if succeeded:
        state.successful_interpreter_count += 1
    state.require_think("python")
    return observation


# Data: canonical action and protocol state. Algorithm: run text-free transitions and emit structured violations.
async def execute_protocol_action(
    action: ParsedAction,
    state: RetoolSkillState,
) -> ActionResult:
    field_name = action.field_name
    content = action.content

    if field_name is None:
        state.used_fields.append("no_action")
        return record_violation(state, "no_action")

    state.used_fields.append(field_name)

    if field_name == "think":
        if not is_substantive_think(content):
            return record_violation(state, "non_substantive_think")
        repeated_think = len(state.used_fields) >= 2 and state.used_fields[-2:] == ["think", "think"]
        state.think_count += 1
        if state.must_think:
            pending_think_source = state.pending_think_source
            state.must_think = False
            state.pending_think_source = None
            state.completed_required_think_count += 1
            if pending_think_source == "skill":
                state.post_skill_think_completed = True
            elif pending_think_source == "python":
                state.post_python_think_completed = True
        if repeated_think:
            state.repeated_think_count += 1
            return record_violation(state, "repeated_think")
        return ActionResult()

    if state.must_think and field_name not in {"skill_body", "interpreter"}:
        return record_violation(
            state,
            f"{field_name}_before_think",
            field_name=field_name,
        )

    if field_name == "skill_call":
        if content != SKILL_NAME:
            return record_violation(
                state,
                f"unsupported_skill_call:{content}",
                requested_skill=content,
                supported_skill=SKILL_NAME,
            )
        if state.skill_loaded:
            return record_violation(state, "duplicate_skill_call")
        state.skill_loaded = True
        state.require_think("skill")
        return ActionResult(observation=build_skill_body_observation())

    if field_name in {"skill_body", "interpreter"}:
        return record_violation(
            state,
            f"forged_{field_name}",
            field_name=field_name,
        )

    if field_name == "tool_call" and not state.skill_loaded:
        return record_violation(
            state,
            f"{field_name}_before_skill_body",
            field_name=field_name,
            skill_name=SKILL_NAME,
        )

    if field_name == "tool_call":
        if state.tool_call_count >= int(TOOL_CONFIGS["max_tool_calls"]):
            return record_violation(state, "too_many_tool_calls")
        code, error = parse_tool_call_json(content)
        if error is not None:
            return record_violation(
                state,
                f"invalid_tool_call:{error}",
                error=error,
            )
        assert code is not None
        return ActionResult(observation=await execute_and_record(code, state))

    if field_name == "answer":
        if state.skill_loaded and state.successful_interpreter_count == 0:
            return record_violation(state, "answer_before_successful_interpreter")
        state.accepted_answer = content
        state.done = True
        return ActionResult(done=True)

    raise AssertionError(f"unsupported action field: {field_name}")
