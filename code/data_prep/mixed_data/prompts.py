# LOCKED: false
"""两部分 system prompt 与统一用户问题构造。"""

from __future__ import annotations

from .contracts import CODE_SKILL, SEARCH_SKILL, SYSTEM_PROMPT_VERSION


GENERIC_SKILL_CALL = "<skill_call>SKILL_NAME</skill_call>"
USER_SKILL_INSTRUCTION = "Select and load the useful skill before reasoning or answering."


# 数据：两个 skill 的名称与短 description。算法：只公开能力目录，不泄漏执行 body。
def build_available_skills_part() -> str:
    return (
        "Part 1: Available skills\n"
        f"- name: {CODE_SKILL.name}\n  purpose: {CODE_SKILL.description}\n"
        f"- name: {SEARCH_SKILL.name}\n  purpose: {SEARCH_SKILL.description}"
    )


# 数据：通用 skill 调用占位符。算法：说明动态加载规则但不给具体可复制调用。
def build_skill_loading_part() -> str:
    return (
        "Part 2: Skill loading\n"
        "Choose the useful skill before reasoning or answering. Output exactly "
        f"{GENERIC_SKILL_CALL} and replace SKILL_NAME with one listed name. "
        "The harness will return its instructions inside <skill_body>...</skill_body>. "
        "Follow the returned instructions and environment observations."
    )


# 数据：两个固定 system prompt 部分。算法：按目录、加载规则顺序拼接并检查不泄漏 body。
def build_system_prompt() -> str:
    prompt = f"{build_available_skills_part()}\n\n{build_skill_loading_part()}"
    assert prompt.count("Part 1:") == 1 and prompt.count("Part 2:") == 1
    assert CODE_SKILL.body not in prompt and SEARCH_SKILL.body not in prompt
    assert f"<skill_call>{CODE_SKILL.name}</skill_call>" not in prompt
    assert f"<skill_call>{SEARCH_SKILL.name}</skill_call>" not in prompt
    assert "<code>" not in prompt and "<search>" not in prompt
    return prompt


# 数据：原始题目文本。算法：压缩空白并保证问题非空，不注入预期任务类型。
def normalize_question(question: str) -> str:
    assert isinstance(question, str), "question must be a string"
    normalized = " ".join(question.strip().split())
    assert normalized, "question must not be empty"
    return normalized


# 数据：规范化题目。算法：仅增加通用 skill-first 要求，不提示应选择哪个 skill。
def build_user_prompt(question: str) -> str:
    return f"Question: {normalize_question(question)}\n\n{USER_SKILL_INSTRUCTION}"


# 数据：单条题目。算法：生成统一的 system/user 两消息 schema。
def build_prompt_messages(question: str) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(question)},
    ]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert SYSTEM_PROMPT_VERSION
    return messages
