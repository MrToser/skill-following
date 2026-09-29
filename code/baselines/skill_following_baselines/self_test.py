# LOCKED: false
"""Deterministic checks for baseline prompt and answer contracts."""

from __future__ import annotations

import json
import os

from slime.utils.types import Sample
from skill_following.config import clear_runtime_config_cache
from skill_following.contracts import CODE_SKILL, SEARCH_SKILL

from .data import direct_prompt, extract_question, preloaded_skill_prompt, retool_message_prompt, search_r1_prompt
from .metrics import _targets_and_task, extract_search_answer
from .preloaded_skill import initialize_preloaded_state
from .source_retool import prompt_text


def main() -> None:
    prompt = [
        {"role": "system", "content": "catalog"},
        {
            "role": "user",
            "content": "Question: What is 2 + 3?\n\nSelect and load the useful skill before reasoning or answering.",
        },
    ]
    assert extract_question(prompt) == "What is 2 + 3?"
    assert "external tools" in direct_prompt("Who?", "search")[0]["content"]
    math_preloaded = preloaded_skill_prompt("2 + 3", "math")
    search_preloaded = preloaded_skill_prompt("Who?", "search")
    assert CODE_SKILL.body in math_preloaded[0]["content"]
    assert SEARCH_SKILL.body in search_preloaded[0]["content"]
    assert "<search>" in search_r1_prompt("Who?")[0]["content"]
    retool_prompt = retool_message_prompt("Compute 2 + 3.")
    assert retool_prompt == [{"role": "user", "content": "Compute 2 + 3."}]
    assert prompt_text(retool_prompt) == "Compute 2 + 3."
    assert prompt_text("Compute 2 + 3.") == "Compute 2 + 3."
    assert extract_search_answer("reason\n<answer>Beijing</answer>") == "Beijing"
    assert extract_search_answer("Final Answer: Paris") == "Paris"
    scalar_task, scalar_targets = _targets_and_task(Sample(label=0, metadata={"task_type": "math"}))
    assert scalar_task == "math" and scalar_targets == ["0"]
    os.environ.update(
        {
            "SF_EXP_ID": "self_test_preloaded_skill",
            "SF_EXP_TASK_MODE": "math",
            "SF_EXP_SKILL_VARIANT": "full",
            "SF_EXP_STATE_AWARE_HINTS": "1",
            "SF_EXP_TRANSITION_CREDIT": "0",
            "SF_EXP_RECOVERY_FACTOR": "0.9",
            "SF_EXP_OBSERVATION_MODE": "normal",
            "SF_EXP_SEED": "20260818",
        }
    )
    clear_runtime_config_cache()
    sample = Sample(label=json.dumps({"task_type": "math", "target": ["5"]}))
    state = initialize_preloaded_state(sample, math_preloaded[0]["content"])
    assert state.loaded_skill == CODE_SKILL.name
    assert state.phase.value == "need_plan"
    assert state.transition_credit_enabled is False
    print(json.dumps({"status": "ok", "tests": 15}, sort_keys=True))


if __name__ == "__main__":
    main()
