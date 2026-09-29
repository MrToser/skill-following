# LOCKED: false
"""Compatibility adapter between processor-backed datasets and source ReTool."""

from __future__ import annotations

from typing import Any

from generate_with_retool import generate as source_generate
from skill_following.logger import (
    my_logger,
)
from slime.utils.types import Sample


LOGGER = my_logger("source-retool-adapter")
_LOGGED_ADAPTATION = False


# 数据：字符串或单轮消息列表 prompt。算法：提取唯一用户文本，使原始 ReTool generator 继续接收其原生字符串接口。
def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        text = prompt.strip()
        assert text, "ReTool prompt must not be empty"
        return text
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    assert isinstance(prompt, list) and prompt, "ReTool message prompt must be a non-empty list"
    user_messages = [message for message in prompt if isinstance(message, dict) and message.get("role") == "user"]
    assert len(user_messages) == 1, "ReTool compatibility view must contain exactly one user message"
    content = user_messages[0].get("content")
    assert isinstance(content, str) and content.strip(), "ReTool user content must be a non-empty string"
    return content.strip()


# 数据：slime Sample 与原始 ReTool generator 参数。算法：在 rollout 边界还原字符串 prompt，其余生成逻辑原样委托。
async def generate(args: Any, sample: Sample, sampling_params: Any) -> Sample:
    global _LOGGED_ADAPTATION
    original_type = type(sample.prompt).__name__
    sample.prompt = prompt_text(sample.prompt)
    if original_type != "str" and not _LOGGED_ADAPTATION:
        LOGGER.info("adapted processor prompt representation from %s to str", original_type)
        _LOGGED_ADAPTATION = True
    return await source_generate(args, sample, sampling_params)


__all__ = ["generate", "prompt_text"]
