# LOCKED: false
"""实验套件的 slime custom generate、reward 与 eval logger 入口。"""

from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from .contracts import RUNTIME_VERSION
from .reward import compute_reward, log_eval_rollout_data, reward_func
from .rollout import generate_unified


VERSION = RUNTIME_VERSION


# 数据：slime generate callback 输入。算法：把统一 reward 预览注入隔离 rollout。
async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    return await generate_unified(
        args,
        sample,
        sampling_params,
        reward_preview_fn=compute_reward,
    )


__all__ = ["VERSION", "generate", "reward_func", "log_eval_rollout_data"]
