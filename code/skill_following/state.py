# LOCKED: false
"""统一 Harness 的显式状态与 transition 记账。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import ProtocolPhase, TRANSITION_NAMES, expected_skill_for_task


@dataclass
class UnifiedSkillState:
    expected_task_type: str
    expected_skill: str
    experiment_id: str
    skill_variant: str
    state_aware_hints: bool
    transition_credit_enabled: bool
    recovery_factor: float
    first_research_reward: float
    observation_mode: str
    phase: ProtocolPhase = ProtocolPhase.NEED_SKILL
    feedback_message_count: int = 0
    feedback_character_count: int = 0
    loaded_skill: str | None = None
    turn_index: int = 0
    used_fields: list[str] = field(default_factory=list)
    accepted_fields: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    transition_recovery: dict[str, bool] = field(default_factory=dict)
    next_action_after_rejected_hint: bool = False
    used_recovery_hint: bool = False
    recovery_hint_count: int = 0
    plan_think_count: int = 0
    evidence_think_count: int = 0
    repeated_think_count: int = 0
    action_count: int = 0
    code_call_count: int = 0
    retrieval_count: int = 0
    observation_count: int = 0
    valid_observation_count: int = 0
    invalid_observation_count: int = 0
    interpreter_count: int = 0
    information_count: int = 0
    summary_count: int = 0
    last_query: str | None = None
    last_observation_valid: bool = False
    premature_answer_count: int = 0
    wrong_skill_count: int = 0
    forged_observation_count: int = 0
    observation_withheld_count: int = 0
    direct_answer_echo_count: int = 0
    action_tail_trimmed_count: int = 0
    discarded_action_tail_token_count: int = 0
    first_accepted_action_turn: int | None = None
    accepted_answer: str | None = None
    injected_skill_source: str | None = None
    injected_skill_body_sha256: str | None = None
    done: bool = False

    # 数据：任务类型和预期 skill。算法：保证数据 contract 与运行时路由一致。
    def __post_init__(self) -> None:
        spec = expected_skill_for_task(self.expected_task_type)
        assert self.expected_skill == spec.name, (
            f"expected_skill mismatch: task={self.expected_task_type}, skill={self.expected_skill}"
        )
        assert self.experiment_id.strip()
        assert self.skill_variant.strip()
        assert 0.0 < self.recovery_factor <= 1.0
        assert 0.0 <= self.first_research_reward <= 0.2
        assert self.observation_mode in {"normal", "withheld"}

    # 数据：一次 rejected hint。算法：仅把下一个被接受 transition 标成 recovery。
    def mark_recovery_hint(self) -> None:
        self.used_recovery_hint = True
        self.recovery_hint_count += 1
        self.next_action_after_rejected_hint = True

    # 数据：一次 accepted hint。算法：关闭局部 recovery 上下文。
    def mark_accepted_hint(self) -> None:
        self.next_action_after_rejected_hint = False

    # 数据：一个 canonical transition。算法：只记录第一次出现及其 recovery 上下文。
    def record_transition(self, transition_name: str) -> None:
        assert transition_name in TRANSITION_NAMES, f"unsupported transition: {transition_name}"
        if transition_name not in self.transition_recovery:
            self.transition_recovery[transition_name] = self.next_action_after_rejected_hint

    # 数据：动作字段与被丢弃 token 数。算法：累计首动作边界诊断。
    def record_action_tail_trim(self, field_name: str, discarded_token_count: int) -> None:
        assert field_name.strip() and discarded_token_count > 0
        self.action_tail_trimmed_count += 1
        self.discarded_action_tail_token_count += discarded_token_count

    # 数据：Harness 实际返回的 skill body 身份。算法：每条轨迹只记录第一次注入。
    def record_injected_skill(self, source_skill: str | None, body_sha256: str) -> None:
        assert len(body_sha256) == 64
        if self.injected_skill_body_sha256 is None:
            self.injected_skill_source = source_skill
            self.injected_skill_body_sha256 = body_sha256
            return
        assert self.injected_skill_source == source_skill
        assert self.injected_skill_body_sha256 == body_sha256

    # 数据：完整运行时状态。算法：转换为 reward 和 trace 可消费的 JSON 安全字典。
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "skill_variant": self.skill_variant,
            "state_aware_hints": self.state_aware_hints,
            "transition_credit_enabled": self.transition_credit_enabled,
            "recovery_factor": self.recovery_factor,
            "first_research_reward": self.first_research_reward,
            "observation_mode": self.observation_mode,
            "expected_task_type": self.expected_task_type,
            "expected_skill": self.expected_skill,
            "protocol_phase": self.phase.value,
            "feedback_message_count": self.feedback_message_count,
            "feedback_character_count": self.feedback_character_count,
            "loaded_skill": self.loaded_skill,
            "skill_loaded": self.loaded_skill is not None,
            "turn_index": self.turn_index,
            "used_fields": list(self.used_fields),
            "accepted_fields": list(self.accepted_fields),
            "violations": list(self.violations),
            "transition_recovery": dict(self.transition_recovery),
            "next_action_after_rejected_hint": self.next_action_after_rejected_hint,
            "used_recovery_hint": self.used_recovery_hint,
            "recovery_hint_count": self.recovery_hint_count,
            "plan_think_count": self.plan_think_count,
            "evidence_think_count": self.evidence_think_count,
            "repeated_think_count": self.repeated_think_count,
            "action_count": self.action_count,
            "code_call_count": self.code_call_count,
            "retrieval_count": self.retrieval_count,
            "observation_count": self.observation_count,
            "valid_observation_count": self.valid_observation_count,
            "invalid_observation_count": self.invalid_observation_count,
            "interpreter_count": self.interpreter_count,
            "information_count": self.information_count,
            "summary_count": self.summary_count,
            "last_query": self.last_query,
            "last_observation_valid": self.last_observation_valid,
            "premature_answer_count": self.premature_answer_count,
            "wrong_skill_count": self.wrong_skill_count,
            "forged_observation_count": self.forged_observation_count,
            "observation_withheld_count": self.observation_withheld_count,
            "direct_answer_echo_count": self.direct_answer_echo_count,
            "action_tail_trimmed_count": self.action_tail_trimmed_count,
            "discarded_action_tail_token_count": self.discarded_action_tail_token_count,
            "first_accepted_action_turn": self.first_accepted_action_turn,
            "accepted_answer": self.accepted_answer,
            "injected_skill_source": self.injected_skill_source,
            "injected_skill_body_sha256": self.injected_skill_body_sha256,
            "done": self.done,
        }
