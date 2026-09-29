# LOCKED: false
"""Protocol state and immutable version policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RetoolSkillState:
    skill_loaded: bool = False
    turn_index: int = 0
    used_fields: list[str] = field(default_factory=list)
    must_think: bool = False
    pending_think_source: str | None = None
    think_count: int = 0
    repeated_think_count: int = 0
    required_think_count: int = 0
    completed_required_think_count: int = 0
    post_skill_think_completed: bool = False
    post_python_think_completed: bool = False
    tool_call_count: int = 0
    interpreter_count: int = 0
    successful_interpreter_count: int = 0
    violations: list[str] = field(default_factory=list)
    accepted_answer: str | None = None
    done: bool = False

    # Data: current state. Algorithm: convert to JSON-safe metadata for reward and trace.
    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_loaded": self.skill_loaded,
            "turn_index": self.turn_index,
            "used_fields": list(self.used_fields),
            "must_think": self.must_think,
            "pending_think_source": self.pending_think_source,
            "think_count": self.think_count,
            "repeated_think_count": self.repeated_think_count,
            "required_think_count": self.required_think_count,
            "completed_required_think_count": self.completed_required_think_count,
            "post_skill_think_completed": self.post_skill_think_completed,
            "post_python_think_completed": self.post_python_think_completed,
            "tool_call_count": self.tool_call_count,
            "interpreter_count": self.interpreter_count,
            "successful_interpreter_count": self.successful_interpreter_count,
            "violations": list(self.violations),
            "accepted_answer": self.accepted_answer,
            "done": self.done,
        }

    # Data: environment source that requires reasoning. Algorithm: open one typed thought gate without overwriting a pending gate.
    def require_think(self, source: str) -> None:
        assert source in {"skill", "python", "violation"}, f"unsupported think source: {source}"
        if not self.must_think:
            self.must_think = True
            self.pending_think_source = source
            self.required_think_count += 1


@dataclass(frozen=True)
class ParsedAction:
    field_name: str | None
    content: str


@dataclass(frozen=True)
class ViolationEvent:
    """Structured base event with no user-facing feedback text."""

    code: str
    context: tuple[tuple[str, str], ...] = ()

    # Data: immutable key/value context. Algorithm: expose a fresh mapping to version-owned handlers.
    def context_dict(self) -> dict[str, str]:
        return dict(self.context)

    # Data: violation code and context. Algorithm: enforce non-empty, unique event metadata.
    def __post_init__(self) -> None:
        assert self.code.strip(), "violation code must not be empty"
        context_keys = [key for key, _ in self.context]
        assert all(key.strip() for key in context_keys), "violation context keys must not be empty"
        assert len(context_keys) == len(set(context_keys)), "violation context contains duplicate keys"


@dataclass(frozen=True)
class ViolationFeedback:
    """Version-owned environment feedback for one base violation event."""

    observation: str
    require_think: bool

    # Data: rendered environment observation. Algorithm: reject empty correction feedback.
    def __post_init__(self) -> None:
        assert self.observation.strip(), "violation feedback observation must not be empty"


@dataclass(frozen=True)
class ActionResult:
    """Result of one base protocol transition before version feedback rendering."""

    observation: str = ""
    violation: ViolationEvent | None = None
    done: bool = False

    # Data: observation, violation, and terminal flag. Algorithm: enforce mutually exclusive result modes.
    def __post_init__(self) -> None:
        assert not (self.observation and self.violation), "result cannot contain observation and violation"
        assert not (self.done and self.violation), "terminal result cannot contain a violation"


@dataclass(frozen=True)
class RetoolRewardConfig:
    """Version-owned non-negative reward weights."""

    stage_reward: float
    tool_success_reward: float
    answer_reward: float
    repeated_think_penalty: float
    max_repeated_think_penalty: float

    # Data: reward weights. Algorithm: enforce non-negative values and a bounded think deduction.
    def __post_init__(self) -> None:
        weights = (
            self.stage_reward,
            self.tool_success_reward,
            self.answer_reward,
            self.repeated_think_penalty,
            self.max_repeated_think_penalty,
        )
        assert all(weight >= 0 for weight in weights), "reward weights must be non-negative"
        assert self.max_repeated_think_penalty <= self.stage_reward, (
            "maximum repeated-think deduction cannot exceed one stage reward"
        )


@dataclass(frozen=True)
class RetoolRuntimeConfig:
    """Version-owned handlers and diagnostics injected into stable rollout algorithms."""

    version: str
    handle_violation: Callable[[ViolationEvent], ViolationFeedback]
    compute_reward: Callable[[Any], dict[str, Any]]
    default_max_turns: int
    trace_reward_fields: tuple[str, ...] | None = None
    trace_skill_state_fields: tuple[str, ...] | None = None
    include_redundant_trace_views: bool = True

    # Data: immutable version policy. Algorithm: reject missing handlers and duplicate trace schemas at import time.
    def __post_init__(self) -> None:
        assert self.version.strip(), "runtime version must not be empty"
        assert callable(self.handle_violation), "runtime violation handler must be callable"
        assert callable(self.compute_reward), "runtime reward function must be callable"
        assert self.default_max_turns > 0, "runtime default max turns must be positive"
        for field_names in (self.trace_reward_fields, self.trace_skill_state_fields):
            if field_names is not None:
                assert field_names, "trace field whitelist must not be empty"
                assert len(field_names) == len(set(field_names)), "trace field whitelist contains duplicates"
