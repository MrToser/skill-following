# LOCKED: false
"""实验条件到运行时行为的唯一配置边界。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


TASK_MODES = {"math", "search", "joint"}
SKILL_VARIANTS = {
    "full",
    "removed",
    "paraphrased",
    "wrong",
}
OBSERVATION_MODES = {"normal", "withheld"}


# 数据：环境变量中的布尔配置。算法：只接受显式真假值，避免拼写错误静默改变实验。
def parse_bool(value: str, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class RuntimeConfig:
    """一条 rollout 中固定不变的干预配置。"""

    experiment_id: str
    task_mode: str
    skill_variant: str
    state_aware_hints: bool
    transition_credit: bool
    recovery_factor: float
    first_research_reward: float
    observation_mode: str
    seed: int

    # 数据：注册表下发的运行时字段。算法：阻止跨任务 skill 消融和非法 reward 系数。
    def __post_init__(self) -> None:
        assert self.experiment_id.strip(), "experiment_id must not be empty"
        assert self.task_mode in TASK_MODES, f"unsupported task_mode: {self.task_mode}"
        assert self.skill_variant in SKILL_VARIANTS, (
            f"unsupported skill_variant: {self.skill_variant}"
        )
        assert self.observation_mode in OBSERVATION_MODES, (
            f"unsupported observation_mode: {self.observation_mode}"
        )
        assert 0.0 < self.recovery_factor <= 1.0
        assert 0.0 <= self.first_research_reward <= 0.2
        assert self.seed >= 0

    # 数据：样本 task_type。算法：检查单任务条件不会混入另一类样本。
    def accepts_task(self, task_type: str) -> bool:
        normalized = str(task_type).strip().lower()
        return self.task_mode == "joint" or self.task_mode == normalized


# 数据：SF_EXP_* 环境变量。算法：构建一次不可变配置并由 worker 进程内缓存。
@lru_cache(maxsize=1)
def get_runtime_config() -> RuntimeConfig:
    recovery_raw = os.environ.get("SF_EXP_RECOVERY_FACTOR", "0.90")
    try:
        recovery_factor = float(recovery_raw)
    except ValueError as exc:
        raise ValueError(
            f"SF_EXP_RECOVERY_FACTOR must be numeric, got {recovery_raw!r}"
        ) from exc
    research_raw = os.environ.get("SF_EXP_FIRST_RESEARCH_REWARD", "0")
    try:
        first_research_reward = float(research_raw)
    except ValueError as exc:
        raise ValueError(
            f"SF_EXP_FIRST_RESEARCH_REWARD must be numeric, got {research_raw!r}"
        ) from exc
    seed_raw = os.environ.get("SF_EXP_SEED", "20260814")
    try:
        seed = int(seed_raw)
    except ValueError as exc:
        raise ValueError(f"SF_EXP_SEED must be an integer, got {seed_raw!r}") from exc
    return RuntimeConfig(
        experiment_id=os.environ.get("SF_EXP_ID", "development_full"),
        task_mode=os.environ.get("SF_EXP_TASK_MODE", "joint").strip().lower(),
        skill_variant=os.environ.get("SF_EXP_SKILL_VARIANT", "full").strip().lower(),
        state_aware_hints=parse_bool(
            os.environ.get("SF_EXP_STATE_AWARE_HINTS", "1"),
            name="SF_EXP_STATE_AWARE_HINTS",
        ),
        transition_credit=parse_bool(
            os.environ.get("SF_EXP_TRANSITION_CREDIT", "1"),
            name="SF_EXP_TRANSITION_CREDIT",
        ),
        recovery_factor=recovery_factor,
        first_research_reward=first_research_reward,
        observation_mode=os.environ.get("SF_EXP_OBSERVATION_MODE", "normal").strip().lower(),
        seed=seed,
    )


# 数据：单元测试临时环境。算法：显式清空进程内配置缓存。
def clear_runtime_config_cache() -> None:
    get_runtime_config.cache_clear()
