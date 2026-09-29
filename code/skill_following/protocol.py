# LOCKED: false
"""统一动作语法、短 hint 与 Search/Code adapter。"""

from __future__ import annotations

import ast
import asyncio
import os
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from .config import get_runtime_config
from .contracts import (
    ACTION_TRANSITION,
    ANSWER_TRANSITION,
    CODE_SKILL_NAME,
    EVIDENCE_THINK_TRANSITION,
    OBSERVATION_TRANSITION,
    PLAN_TRANSITION,
    RESEARCH_TRANSITION,
    ProtocolPhase,
    SEARCH_SKILL_NAME,
    SKILL_SPECS,
    SKILL_TRANSITION,
    TaskType,
    SUMMARY_TRANSITION,
    injected_skill_body,
)
from .logger import my_logger
from .python_sandbox import execute_python
from .retool_base.protocol import is_substantive_think
from .state import UnifiedSkillState


LOGGER = my_logger("skill_following_experiment_suite.protocol")
MAX_HINT_CHARS = 280
MAX_TOOL_ACTIONS = 8
MAX_QUERY_CHARS = 180
MAX_QUERY_WORDS = 28
MIN_SUMMARY_CHARS = 24
MIN_SUMMARY_TOKENS = 4
STOP_TAGS = (
    "</skill_call>",
    "</think>",
    "</code>",
    "</search>",
    "</research>",
    "</summary>",
    "</answer>",
    "</tool_call>",
)
ACTION_PATTERN = re.compile(
    r"<(skill_call|think|code|search|research|summary|answer|tool_call|skill_body|interpreter|information)>(.*?)</\1>",
    re.DOTALL,
)
BOXED_ANSWER_PATTERN = re.compile(
    r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}",
    re.DOTALL,
)
INTERPRETER_PATTERN = re.compile(r"<interpreter>\s*(.*?)\s*</interpreter>", re.DOTALL)
_SEARCH_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("UNIFIED_SEARCH_CONCURRENCY", "64")))


@dataclass(frozen=True)
class ParsedAction:
    field_name: str | None
    content: str


@dataclass(frozen=True)
class EnvironmentResult:
    observation: str
    valid: bool

    # 数据：环境 observation 与有效性。算法：保证真实环境总是返回非空文本。
    def __post_init__(self) -> None:
        assert self.observation.strip(), "environment observation must not be empty"


@dataclass(frozen=True)
class ActionResult:
    observation: str = ""
    done: bool = False
    accepted: bool = False


EnvironmentExecutor = Callable[[str, str], Awaitable[EnvironmentResult]]


# 数据：当前状态与上一动作接受情况。算法：用选择后的 feedback phase 生成 adapter-aware 三段式短 hint。
def build_phase_hint(state: UnifiedSkillState, *, accepted: bool) -> str:
    prefix = "Action accepted." if accepted else "Action rejected."
    phase = state.phase
    if phase == ProtocolPhase.NEED_SKILL:
        return f"{prefix} State: no skill is loaded. Next: select one listed skill with <skill_call>...</skill_call>."
    if phase == ProtocolPhase.NEED_PLAN:
        return f"{prefix} State: the skill is loaded. Next: write one complete <think>...</think> plan."
    if phase == ProtocolPhase.NEED_EVIDENCE_THINK:
        return f"{prefix} State: an environment observation is ready. Next: write one complete <think>...</think>."

    spec = SKILL_SPECS[state.expected_skill]
    if phase == ProtocolPhase.NEED_ACTION:
        action_tag = spec.first_action if state.action_count == 0 else spec.repeated_action
        if spec.task_type == TaskType.MATH:
            return f"{prefix} State: Python evidence is required. Next: run calculation in <{action_tag}>...</{action_tag}> and print a result."
        return f"{prefix} State: retrieval evidence is required. Next: use a concise <{action_tag}>...</{action_tag}> query."

    assert phase == ProtocolPhase.CAN_CONTINUE_OR_ANSWER
    if spec.task_type == TaskType.MATH:
        return f"{prefix} State: evidence is analyzed. Next: use <code>...</code> again or give Answer: \\boxed{{...}}."
    if state.summary_count > 0:
        return f"{prefix} State: evidence is summarized. Next: use <research>...</research> again or give <answer>...</answer>."
    return f"{prefix} State: evidence is analyzed. Next: use <research>...</research> again, organize it with <summary>...</summary>, or give <answer>...</answer>."


# 数据：三段式 hint 文本。算法：归一化空白、限制长度并用环境标签包裹。
def wrap_hint(message: str) -> str:
    normalized = " ".join(message.strip().split())
    assert normalized
    assert "accepted." in normalized.lower() or "rejected." in normalized.lower()
    assert "State:" in normalized and "Next:" in normalized
    assert ";" not in normalized and "；" not in normalized
    assert len(normalized) <= MAX_HINT_CHARS, f"hint too long: {len(normalized)}"
    return f"\n\n<hint>{normalized}</hint>\n\n"


# 数据：当前状态与动作接受情况。算法：更新 recovery 节奏并返回对应短 hint。
def phase_hint_observation(state: UnifiedSkillState, *, accepted: bool) -> str:
    if not state.state_aware_hints:
        state.mark_accepted_hint()
        return ""
    if accepted:
        state.mark_accepted_hint()
    else:
        state.mark_recovery_hint()
    message = wrap_hint(build_phase_hint(state, accepted=accepted))
    state.feedback_message_count += 1
    state.feedback_character_count += len(message)
    return message


# 数据：模板预先打开的首轮 think 和当前状态。算法：给出不计为拒绝或恢复的继续提示。
def prefilled_think_hint_observation(state: UnifiedSkillState) -> str:
    assert state.phase == ProtocolPhase.NEED_SKILL
    if not state.state_aware_hints:
        return ""
    message = (
        "\n\n<hint>Ready. State: no skill is loaded. Next: select one listed skill "
        "with <skill_call>...</skill_call>.</hint>\n\n"
    )
    state.feedback_message_count += 1
    state.feedback_character_count += len(message)
    return message


# 数据：skill 名称与轨迹状态。算法：按实验条件注入 body 并记录内容哈希。
def build_skill_body_observation(skill_name: str, state: UnifiedSkillState) -> str:
    assert skill_name in SKILL_SPECS
    injected = injected_skill_body(skill_name)
    state.record_injected_skill(injected.source_skill_name, injected.sha256)
    return f"\n\n<skill_body>{injected.body}</skill_body>\n\n"


# 数据：真实环境结果与当前任务。算法：仅替换 observation payload，保留执行有效性。
def apply_observation_intervention(
    result: EnvironmentResult,
    state: UnifiedSkillState,
) -> EnvironmentResult:
    runtime = get_runtime_config()
    assert runtime.observation_mode == state.observation_mode
    if runtime.observation_mode == "normal" or not result.valid:
        return result
    state.observation_withheld_count += 1
    if state.expected_task_type == TaskType.MATH.value:
        observation = "\n\n<interpreter>Output unavailable.</interpreter>\n\n"
    else:
        observation = "\n\n<information>No evidence available.</information>\n\n"
    return EnvironmentResult(observation=observation, valid=True)


# 数据：原 sampling 参数。算法：合并所有完整动作闭合标签并保持调用方配置。
def with_protocol_stop_tags(sampling_params: dict) -> dict:
    existing_stop = sampling_params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    return {**sampling_params, "stop": list(dict.fromkeys([*existing_stop, *STOP_TAGS]))}


# 数据：单轮模型文本。算法：只接受从 turn 起点开始的第一个完整动作。
def parse_action(prediction: str) -> ParsedAction:
    assert isinstance(prediction, str)
    stripped = prediction.strip()
    tagged_match = ACTION_PATTERN.match(stripped)
    boxed_match = BOXED_ANSWER_PATTERN.match(stripped)
    if tagged_match is None and boxed_match is None:
        return ParsedAction(field_name=None, content="")
    if boxed_match is not None and (tagged_match is None or boxed_match.start() < tagged_match.start()):
        return ParsedAction(field_name="boxed_answer", content=boxed_match.group(1).strip())
    assert tagged_match is not None
    return ParsedAction(field_name=tagged_match.group(1), content=tagged_match.group(2).strip())


# 数据：渲染后的 prompt 和首轮生成文本。算法：只识别模板打开、模型关闭的首个 think。
def is_initial_prefilled_think(prompt_text: str, generated_text: str) -> bool:
    return prompt_text.rstrip().endswith("<think>") and generated_text.rstrip().endswith("</think>")


# 数据：Python AST。算法：查找直接 built-in print 调用，保证环境证据可见。
def contains_print_call(module: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(module)
    )


# 数据：code 标签内文本。算法：要求非空、语法合法并调用 print。
def parse_python_code(content: str) -> tuple[str | None, str | None]:
    code = content.strip()
    if not code:
        return None, "empty_code"
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None, "invalid_python_syntax"
    if not contains_print_call(module):
        return None, "missing_print"
    return code, None


# 数据：Python AST node。算法：识别正负号包裹的直接 literal。
def is_literal_value(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) or (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
    )


# 数据：合法 Python 源码。算法：识别只打印硬编码答案的无计算捷径。
def is_direct_answer_echo(code: str) -> bool:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return False
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
        return False
    expression = module.body[0].value
    return (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "print"
        and bool(expression.args)
        and all(is_literal_value(argument) for argument in expression.args)
        and not expression.keywords
    )


# 数据：interpreter observation。算法：去掉环境标签与 Output 前缀，提取可见 stdout。
def extract_visible_code_output(observation: str) -> str:
    match = INTERPRETER_PATTERN.search(observation)
    if match is None:
        return ""
    output = match.group(1).strip()
    if output.startswith("Output:"):
        output = output[len("Output:") :].strip()
    return output


# 数据：检索 query。算法：用长度、词数、标签字符和有效字符过滤不可执行请求。
def is_reasonable_search_query(query: str) -> bool:
    normalized = " ".join(query.strip().split())
    if not normalized or len(normalized) > MAX_QUERY_CHARS:
        return False
    if "<" in normalized or ">" in normalized:
        return False
    if len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", normalized)) > MAX_QUERY_WORDS:
        return False
    return sum(character.isalnum() for character in normalized) >= 3


# 数据：检索 query。算法：用小写字母数字 token 做重复判断。
def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", query.lower()))


# 数据：summary 文本。算法：要求足够长且包含多个有效 token，拒绝空壳总结骗取过程分。
def is_substantive_summary(content: str) -> bool:
    normalized = " ".join(content.strip().split())
    if len(normalized) < MIN_SUMMARY_CHARS or "<" in normalized or ">" in normalized:
        return False
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*|[\u4e00-\u9fff]", normalized)
    return len(tokens) >= MIN_SUMMARY_TOKENS


# 数据：Python 源码。算法：调用稳定 sandbox 并仅把成功且非空 stdout 视为有效 observation。
async def execute_code_action(code: str) -> EnvironmentResult:
    observation, succeeded = await execute_python(code)
    visible_output = extract_visible_code_output(observation)
    return EnvironmentResult(observation=observation, valid=succeeded and bool(visible_output))


# 数据：检索返回文档列表。算法：压缩成 Search-R1 兼容的编号 information 文本。
def format_search_results(results: list[dict]) -> str:
    formatted: list[str] = []
    for index, item in enumerate(results, start=1):
        document = item.get("document") if isinstance(item, dict) else None
        contents = document.get("contents", "") if isinstance(document, dict) else ""
        contents = str(contents).strip()
        if contents:
            formatted.append(f"Doc {index}: {contents}")
    return "\n".join(formatted)


# 数据：检索 query 与运行时 URL。算法：调用本地 retriever，并把空结果保留为可推理的无效 observation。
async def execute_search_action(query: str) -> EnvironmentResult:
    from local_search_server import local_search

    search_url = os.environ.get("UNIFIED_SEARCH_URL", "http://127.0.0.1:8000/retrieve").strip()
    top_k = int(os.environ.get("UNIFIED_SEARCH_TOPK", "3"))
    assert search_url and top_k > 0
    try:
        async with _SEARCH_SEMAPHORE:
            results = await local_search(search_url, query, top_k=top_k)
        formatted = format_search_results(results)
    except Exception as exc:  # 环境失败应进入轨迹，由模型下一轮修正。
        LOGGER.warning("search failed for query=%r: %s", query, exc)
        formatted = ""
    if not formatted:
        return EnvironmentResult(
            observation="\n\n<information>No search results were returned.</information>\n\n",
            valid=False,
        )
    return EnvironmentResult(
        observation=f"\n\n<information>{formatted}</information>\n\n",
        valid=True,
    )


# 数据：canonical action 名称与 payload。算法：只在 adapter 边界分派 Python 或检索环境。
async def execute_environment_action(field_name: str, content: str) -> EnvironmentResult:
    if field_name == "code":
        return await execute_code_action(content)
    assert field_name in {"search", "research"}
    return await execute_search_action(content)


# 数据：当前状态与违规码。算法：保持 phase 不变、记录 recovery，并给出同一状态的下一步。
def reject_action(state: UnifiedSkillState, violation: str) -> ActionResult:
    assert violation.strip()
    state.violations.append(violation)
    return ActionResult(
        observation=phase_hint_observation(state, accepted=False),
        accepted=False,
    )


# 数据：解析动作、统一状态和可替换环境 executor。算法：执行闭环状态迁移并生成 loss-masked observation。
async def execute_protocol_action(
    action: ParsedAction,
    state: UnifiedSkillState,
    *,
    environment_executor: EnvironmentExecutor = execute_environment_action,
) -> ActionResult:
    field_name = action.field_name
    content = action.content
    if field_name is None:
        state.used_fields.append("no_action")
        return reject_action(state, "no_action")

    state.used_fields.append(field_name)

    if field_name == "skill_call":
        if state.phase != ProtocolPhase.NEED_SKILL:
            return reject_action(state, "skill_call_out_of_phase")
        if content not in SKILL_SPECS:
            return reject_action(state, "unsupported_skill_call")
        if content != state.expected_skill:
            state.wrong_skill_count += 1
            return reject_action(state, "wrong_skill_for_task")
        state.record_transition(SKILL_TRANSITION)
        state.loaded_skill = content
        state.accepted_fields.append("skill_call")
        state.phase = ProtocolPhase.NEED_PLAN
        return ActionResult(
            observation=build_skill_body_observation(content, state)
            + phase_hint_observation(state, accepted=True),
            accepted=True,
        )

    if field_name in {"skill_body", "interpreter", "information"}:
        state.forged_observation_count += 1
        return reject_action(state, f"forged_{field_name}")

    if field_name == "think":
        if not is_substantive_think(content):
            return reject_action(state, "non_substantive_think")
        if state.phase == ProtocolPhase.NEED_PLAN:
            state.record_transition(PLAN_TRANSITION)
            state.plan_think_count += 1
            state.accepted_fields.append("think")
            state.phase = ProtocolPhase.NEED_ACTION
            return ActionResult(
                observation=phase_hint_observation(state, accepted=True),
                accepted=True,
            )
        if state.phase == ProtocolPhase.NEED_EVIDENCE_THINK:
            state.record_transition(EVIDENCE_THINK_TRANSITION)
            state.evidence_think_count += 1
            state.accepted_fields.append("think")
            state.phase = (
                ProtocolPhase.CAN_CONTINUE_OR_ANSWER
                if state.last_observation_valid
                else ProtocolPhase.NEED_ACTION
            )
            return ActionResult(
                observation=phase_hint_observation(state, accepted=True),
                accepted=True,
            )
        state.repeated_think_count += 1
        return reject_action(state, "think_out_of_phase")

    if field_name == "tool_call":
        return reject_action(state, "legacy_tool_call")

    if field_name in {"code", "search", "research"}:
        if state.phase not in {ProtocolPhase.NEED_ACTION, ProtocolPhase.CAN_CONTINUE_OR_ANSWER}:
            return reject_action(state, "tool_action_out_of_phase")
        if state.action_count >= MAX_TOOL_ACTIONS:
            return reject_action(state, "too_many_tool_actions")

        spec = SKILL_SPECS[state.expected_skill]
        allowed_action = spec.first_action if state.action_count == 0 else spec.repeated_action
        if field_name != allowed_action:
            return reject_action(state, f"expected_{allowed_action}_action")

        executable_content = content.strip()
        if field_name == "code":
            code, error = parse_python_code(executable_content)
            if error is not None:
                return reject_action(state, error)
            assert code is not None
            if is_direct_answer_echo(code):
                state.direct_answer_echo_count += 1
                return reject_action(state, "direct_answer_echo_code")
            executable_content = code
        else:
            if not is_reasonable_search_query(executable_content):
                return reject_action(state, "invalid_search_query")
            if state.last_query and normalize_query(executable_content) == normalize_query(state.last_query):
                return reject_action(state, "repeated_search_query")

        state.record_transition(ACTION_TRANSITION)
        environment_result = await environment_executor(field_name, executable_content)
        environment_result = apply_observation_intervention(environment_result, state)
        if state.first_accepted_action_turn is None:
            state.first_accepted_action_turn = state.turn_index
        state.action_count += 1
        state.observation_count += 1
        state.last_observation_valid = environment_result.valid
        state.accepted_fields.append(field_name)
        if field_name == "code":
            state.code_call_count += 1
            state.interpreter_count += 1
        else:
            state.retrieval_count += 1
            state.information_count += 1
            state.last_query = executable_content
        if environment_result.valid:
            state.valid_observation_count += 1
            state.record_transition(OBSERVATION_TRANSITION)
            if field_name == "research":
                state.record_transition(RESEARCH_TRANSITION)
        else:
            state.invalid_observation_count += 1
        state.phase = ProtocolPhase.NEED_EVIDENCE_THINK
        return ActionResult(
            observation=environment_result.observation
            + phase_hint_observation(state, accepted=True),
            accepted=True,
        )

    if field_name == "summary":
        if state.expected_task_type != TaskType.SEARCH.value:
            return reject_action(state, "summary_not_supported_for_math")
        if state.phase != ProtocolPhase.CAN_CONTINUE_OR_ANSWER:
            return reject_action(state, "summary_out_of_phase")
        if state.summary_count > 0:
            return reject_action(state, "duplicate_summary")
        if not is_substantive_summary(content):
            return reject_action(state, "non_substantive_summary")
        state.record_transition(SUMMARY_TRANSITION)
        state.summary_count += 1
        state.accepted_fields.append("summary")
        return ActionResult(
            observation=phase_hint_observation(state, accepted=True),
            accepted=True,
        )

    if field_name in {"answer", "boxed_answer"}:
        expected_answer_field = (
            "boxed_answer" if state.expected_task_type == TaskType.MATH.value else "answer"
        )
        if field_name != expected_answer_field:
            return reject_action(state, "wrong_answer_format")
        if state.phase != ProtocolPhase.CAN_CONTINUE_OR_ANSWER:
            state.premature_answer_count += 1
            return reject_action(state, "answer_out_of_phase")
        if not content.strip():
            return reject_action(state, "empty_answer")
        state.record_transition(ANSWER_TRANSITION)
        state.mark_accepted_hint()
        state.accepted_fields.append("answer")
        state.accepted_answer = content.strip()
        state.done = True
        return ActionResult(done=True, accepted=True)

    raise AssertionError(f"unsupported action field: {field_name}")
