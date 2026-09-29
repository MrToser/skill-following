# LOCKED: false
"""Search/Code 共用协议、skill 与标签契约。"""

from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import RuntimeConfig, get_runtime_config


CODE_SKILL_NAME = "code-interpreter-protocol"
SEARCH_SKILL_NAME = "qa-search-protocol"
SYSTEM_PROMPT_VERSION = "skill_following_prompt"
RUNTIME_VERSION = "skill_following"

SKILL_TRANSITION = "skill"
PLAN_TRANSITION = "plan"
ACTION_TRANSITION = "action"
OBSERVATION_TRANSITION = "observation"
EVIDENCE_THINK_TRANSITION = "evidence_think"
ANSWER_TRANSITION = "answer"
SUMMARY_TRANSITION = "summary"
RESEARCH_TRANSITION = "research"
CORE_TRANSITION_NAMES = (
    SKILL_TRANSITION,
    PLAN_TRANSITION,
    ACTION_TRANSITION,
    OBSERVATION_TRANSITION,
    EVIDENCE_THINK_TRANSITION,
    ANSWER_TRANSITION,
)
TRANSITION_NAMES = (*CORE_TRANSITION_NAMES, SUMMARY_TRANSITION, RESEARCH_TRANSITION)


class TaskType(str, Enum):
    MATH = "math"
    SEARCH = "search"


class ProtocolPhase(str, Enum):
    NEED_SKILL = "need_skill"
    NEED_PLAN = "need_plan"
    NEED_ACTION = "need_action"
    NEED_EVIDENCE_THINK = "need_evidence_think"
    CAN_CONTINUE_OR_ANSWER = "can_continue_or_answer"


@dataclass(frozen=True)
class SkillSpec:
    """模型可见 skill 与环境 adapter 的最小不可变契约。"""

    name: str
    task_type: TaskType
    description: str
    body: str
    first_action: str
    repeated_action: str
    observation_tag: str
    answer_format: str
    environment_adapter: str
    reward_source: str

    # 数据：skill 字段。算法：在模块加载时阻止 body、adapter 与任务类型错配。
    def __post_init__(self) -> None:
        assert self.name.strip() and self.description.strip() and self.body.strip()
        assert self.first_action in {"code", "search"}
        assert self.repeated_action in {"code", "research"}
        assert self.observation_tag in {"interpreter", "information"}
        assert "<think>...</think>" in self.body
        assert f"<{self.first_action}>" in self.body
        assert f"<{self.observation_tag}>" in self.body
        assert ";" not in self.body and "；" not in self.body


CODE_SKILL = SkillSpec(
    name=CODE_SKILL_NAME,
    task_type=TaskType.MATH,
    description=(
        "Use this skill for mathematical problems that need reliable calculation, "
        "enumeration, symbolic checking, or program-aided verification."
    ),
    body=(
        "Use Python calculations as evidence for the mathematical solution. "
        "First write one complete <think>...</think> to plan the calculation. "
        "Then put executable Python inside <code>...</code>. The code must perform "
        "the calculation and print a result. Wait for the environment-owned "
        "<interpreter>...</interpreter> observation and never write that tag yourself. "
        "Write another complete <think>...</think> to interpret the result. Repeat "
        "<code>...</code> and result reasoning when more calculation is useful. "
        "Finish with Answer: \\boxed{...} only after analyzing a valid result."
    ),
    first_action="code",
    repeated_action="code",
    observation_tag="interpreter",
    answer_format="Answer: \\boxed{...}",
    environment_adapter="python_sandbox",
    reward_source="math_dapo",
)

SEARCH_SKILL = SkillSpec(
    name=SEARCH_SKILL_NAME,
    task_type=TaskType.SEARCH,
    description=(
        "Use this skill for factual questions that need external evidence or "
        "multi-hop verification."
    ),
    body=(
        "Use retrieved evidence to answer the factual question. First write one complete "
        "<think>...</think> to plan the retrieval. Use <search>...</search> for the first "
        "concise query. Wait for the environment-owned <information>...</information> "
        "observation and never write that tag yourself. Write another complete "
        "<think>...</think> to assess the evidence. If more evidence is needed, use a "
        "refined <research>...</research> query and repeat result reasoning. An optional "
        "<summary>...</summary> may organize gathered evidence but does not replace "
        "reasoning. Finish with <answer>...</answer> only after analyzing valid evidence."
    ),
    first_action="search",
    repeated_action="research",
    observation_tag="information",
    answer_format="<answer>...</answer>",
    environment_adapter="dense_retriever",
    reward_source="qa_em",
)

CODE_PARAPHRASED_BODY = (
    "Solve the mathematical problem with evidence produced by Python. Begin with one "
    "complete <think>...</think> that identifies the needed calculation. Place executable "
    "Python in <code>...</code> and print the computed value. Only the environment may "
    "return <interpreter>...</interpreter>. After each result, write a complete "
    "<think>...</think> that interprets it. Alternate further code and reasoning when "
    "needed. Give Answer: \\boxed{...} after using a valid result."
)
SEARCH_PARAPHRASED_BODY = (
    "Answer the factual question from retrieved evidence. Start with one complete "
    "<think>...</think> that identifies what to retrieve. Put the first focused query in "
    "<search>...</search>. Only the environment may return "
    "<information>...</information>. After each observation, write a complete "
    "<think>...</think> that evaluates its relevance. Use a revised "
    "<research>...</research> query when evidence is insufficient. A "
    "<summary>...</summary> may organize evidence. Finish with <answer>...</answer> only "
    "after evidence analysis."
)

SKILL_SPECS = {
    CODE_SKILL.name: CODE_SKILL,
    SEARCH_SKILL.name: SEARCH_SKILL,
}
TASK_SKILLS = {
    TaskType.MATH.value: CODE_SKILL,
    TaskType.SEARCH.value: SEARCH_SKILL,
}


@dataclass(frozen=True)
class InjectedSkillBody:
    """Harness 实际注入的 skill body 及其审计身份。"""

    requested_skill_name: str
    source_skill_name: str | None
    variant: str
    body: str
    sha256: str


# 数据：task mode 与两种固定 skill。算法：只向单任务条件展示对应 skill。
def skill_specs_for_mode(task_mode: str) -> tuple[SkillSpec, ...]:
    normalized = str(task_mode).strip().lower()
    if normalized == TaskType.MATH.value:
        return (CODE_SKILL,)
    if normalized == TaskType.SEARCH.value:
        return (SEARCH_SKILL,)
    assert normalized == "joint", f"unsupported task mode: {task_mode}"
    return (CODE_SKILL, SEARCH_SKILL)


# 数据：被调用的 skill 与实验干预。算法：只改变 body，不改变 wrapper、parser 或状态机。
def injected_skill_body(
    skill_name: str,
    runtime: RuntimeConfig | None = None,
) -> InjectedSkillBody:
    assert skill_name in SKILL_SPECS, f"unsupported skill: {skill_name}"
    runtime = runtime or get_runtime_config()
    requested = SKILL_SPECS[skill_name]
    source_skill_name: str | None = skill_name
    if runtime.skill_variant == "full":
        body = requested.body
    elif runtime.skill_variant == "removed":
        body = ""
        source_skill_name = None
    elif runtime.skill_variant == "paraphrased":
        body = CODE_PARAPHRASED_BODY if skill_name == CODE_SKILL_NAME else SEARCH_PARAPHRASED_BODY
    else:
        assert runtime.skill_variant == "wrong"
        source = SEARCH_SKILL if skill_name == CODE_SKILL_NAME else CODE_SKILL
        body = source.body
        source_skill_name = source.name
    return InjectedSkillBody(
        requested_skill_name=skill_name,
        source_skill_name=source_skill_name,
        variant=runtime.skill_variant,
        body=body,
        sha256=sha256(body.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class TrainingLabel:
    """混合数据中统一的任务标签。"""

    task_type: str
    targets: tuple[str, ...]


# 数据：任务类型字符串。算法：映射到唯一预期 skill，避免业务分支散落。
def expected_skill_for_task(task_type: str) -> SkillSpec:
    normalized = str(task_type).strip().lower()
    assert normalized in TASK_SKILLS, f"unsupported task_type: {task_type}"
    return TASK_SKILLS[normalized]


# 数据：JSON 字符串或字典形式的统一 label。算法：解析并规范成非空 target 元组。
def parse_training_label(label: Any) -> TrainingLabel:
    if isinstance(label, str):
        payload = json.loads(label)
    else:
        payload = label
    assert isinstance(payload, dict), "unified label must be a JSON object"
    task_type = str(payload.get("task_type") or "").strip().lower()
    expected_skill_for_task(task_type)
    raw_targets = payload.get("target")
    if hasattr(raw_targets, "tolist"):
        raw_targets = raw_targets.tolist()
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    assert isinstance(raw_targets, (list, tuple)), "label target must be a list"
    targets = tuple(str(target).strip() for target in raw_targets if str(target).strip())
    assert targets, "label targets must not be empty"
    return TrainingLabel(task_type=task_type, targets=targets)
