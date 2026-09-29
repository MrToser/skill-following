# LOCKED: false
"""Shared immutable constants for the Retool runtime."""

from __future__ import annotations

import re
from pathlib import Path


SKILL_NAME = "code-interpreter-protocol"
DEFAULT_SKILL_PATH = Path("docs/skills/code_interpreter_protocol.md")
SECTION_PATTERN = re.compile(
    r"<!-- skill:(description|skill_body):start -->\s*(.*?)\s*<!-- skill:\1:end -->",
    re.DOTALL,
)
ACTION_PATTERN = re.compile(
    r"<(think|skill_call|tool_call|skill_body|interpreter)>(.*?)</\1>",
    re.DOTALL,
)
ANSWER_PATTERN = re.compile(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
STOP_TAGS = ["</think>", "</skill_call>", "</tool_call>"]
MIN_SUBSTANTIVE_THINK_CHARS = 12
TRACE_TRAIN_TOKENS_ENV = "RETOOL_SKILL_TRACE_TRAIN_TOKENS"
TRACE_MAX_SAMPLES_ENV = "RETOOL_SKILL_TRACE_MAX_SAMPLES"
TRACE_MAX_CHARS_ENV = "RETOOL_SKILL_TRACE_MAX_CHARS"
TRACE_TOKEN_IDS_ENV = "RETOOL_SKILL_TRACE_TOKEN_IDS"
TRACE_OUTPUT_PATH_ENV = "RETOOL_SKILL_TRACE_OUTPUT_PATH"
DEFAULT_TRACE_MAX_SAMPLES = 5
DEFAULT_TRACE_MAX_CHARS = 12000
