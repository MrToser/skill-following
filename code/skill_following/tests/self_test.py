# LOCKED: false
"""实验干预、状态机、reward 与注册表的确定性 CPU 自测。"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
from slime.utils.types import Sample
from slime.utils.eval_config import build_eval_dataset_configs, ensure_dataset_list

from ..config import clear_runtime_config_cache, get_runtime_config
from ..contracts import (
    CODE_SKILL_NAME,
    SEARCH_SKILL_NAME,
    ProtocolPhase,
    injected_skill_body,
)
from ..prompts import build_system_prompt
from ..protocol import (
    EnvironmentResult,
    ParsedAction,
    execute_protocol_action,
    is_initial_prefilled_think,
    prefilled_think_hint_observation,
)
from ..registry import SUITE_ROOT, build_environment, load_registry, resolve_experiment
from ..reporting.prefill_audit import audit_file, has_prefill_rejection
from ..reward import compute_eval_metrics, compute_reward
from ..state import UnifiedSkillState


# 数据：实验配置覆盖。算法：测试结束后恢复环境并清理缓存。
@contextmanager
def runtime_environment(**overrides: str):
    values = {
        "SF_EXP_ID": "self_test",
        "SF_EXP_TASK_MODE": "joint",
        "SF_EXP_SKILL_VARIANT": "full",
        "SF_EXP_STATE_AWARE_HINTS": "1",
        "SF_EXP_TRANSITION_CREDIT": "1",
        "SF_EXP_RECOVERY_FACTOR": "0.90",
        "SF_EXP_FIRST_RESEARCH_REWARD": "0",
        "SF_EXP_OBSERVATION_MODE": "normal",
        "SF_EXP_SEED": "20260814",
        **overrides,
    }
    with patch.dict(os.environ, values, clear=False):
        clear_runtime_config_cache()
        yield get_runtime_config()
    clear_runtime_config_cache()


# 数据：task_type 与当前运行配置。算法：构造所有审计字段一致的状态。
def build_state(task_type: str) -> UnifiedSkillState:
    runtime = get_runtime_config()
    expected_skill = CODE_SKILL_NAME if task_type == "math" else SEARCH_SKILL_NAME
    return UnifiedSkillState(
        expected_task_type=task_type,
        expected_skill=expected_skill,
        experiment_id=runtime.experiment_id,
        skill_variant=runtime.skill_variant,
        state_aware_hints=runtime.state_aware_hints,
        transition_credit_enabled=runtime.transition_credit,
        recovery_factor=runtime.recovery_factor,
        first_research_reward=runtime.first_research_reward,
        observation_mode=runtime.observation_mode,
    )


# 数据：动作类型与 payload。算法：返回确定性的有效 Math/Search observation。
async def fake_environment(field_name: str, content: str) -> EnvironmentResult:
    assert content.strip()
    if field_name == "code":
        return EnvironmentResult(
            observation="\n\n<interpreter>Output: 42</interpreter>\n\n",
            valid=True,
        )
    assert field_name in {"search", "research"}
    return EnvironmentResult(
        observation=(
            "\n\n<information>Doc 1: Hamlet was written by William Shakespeare."
            "</information>\n\n"
        ),
        valid=True,
    )


# 数据：统一 label 和状态。算法：构造 reward callback 可消费的完成或截断 Sample。
def reward_sample(task_type: str, target: str, state: UnifiedSkillState) -> Sample:
    return Sample(
        prompt="Question: self test",
        response="self test trajectory",
        label=json.dumps({"task_type": task_type, "target": [target]}),
        status=Sample.Status.COMPLETED if state.done else Sample.Status.TRUNCATED,
        metadata={"skill_protocol_state": state.to_dict()},
    )


# 数据：Search 完整轨迹及可选 summary。算法：检查主运行时的提示、首次里程碑分和可选终止行为。
async def test_search_summary() -> None:
    with runtime_environment(SF_EXP_TASK_MODE="search"):
        for include_summary, expected_score in ((False, 1.9), (True, 2.0)):
            state = build_state("search")
            actions = [
                ParsedAction("skill_call", SEARCH_SKILL_NAME),
                ParsedAction("think", "I will retrieve the author and verify the evidence."),
                ParsedAction("search", "Hamlet author"),
                ParsedAction("think", "The retrieved document identifies the author of Hamlet."),
            ]
            if include_summary:
                actions.append(ParsedAction("summary", "The retrieved evidence states that William Shakespeare wrote Hamlet."))
            actions.append(ParsedAction("answer", "William Shakespeare"))
            results = []
            for action in actions:
                result = await execute_protocol_action(action, state, environment_executor=fake_environment)
                assert result.accepted
                results.append(result)
            assert "<summary>...</summary>" in results[3].observation
            if include_summary:
                assert "evidence is summarized" in results[4].observation
            reward = compute_reward(reward_sample("search", "William Shakespeare", state))
            assert reward["score"] == expected_score
            assert reward["summary_transition_reward"] == (0.1 if include_summary else 0.0)
            assert reward["summary_completed"] == include_summary
            assert reward["protocol_completed"]

        state = build_state("search")
        for action in (
            ParsedAction("skill_call", SEARCH_SKILL_NAME),
            ParsedAction("think", "I will retrieve the author and verify the evidence."),
            ParsedAction("search", "Hamlet author"),
            ParsedAction("think", "The retrieved document identifies the author of Hamlet."),
        ):
            assert (await execute_protocol_action(action, state, environment_executor=fake_environment)).accepted
        rejected = await execute_protocol_action(ParsedAction("summary", "Shakespeare"), state, environment_executor=fake_environment)
        assert not rejected.accepted
        assert state.violations[-1] == "non_substantive_summary"
        accepted = await execute_protocol_action(
            ParsedAction("summary", "The retrieved evidence states that William Shakespeare wrote Hamlet."),
            state,
            environment_executor=fake_environment,
        )
        assert accepted.accepted
        reward = compute_reward(reward_sample("search", "William Shakespeare", state))
        assert reward["summary_transition_reward"] == 0.09


# 数据：首次有效 research 与重复 research。算法：新实验单次加分，原论文设置维持零分。
async def test_first_research_reward() -> None:
    actions = (
        ParsedAction("skill_call", SEARCH_SKILL_NAME),
        ParsedAction("think", "I will retrieve the author and verify the evidence."),
        ParsedAction("search", "Hamlet author"),
        ParsedAction("think", "The first result identifies Shakespeare; I will verify it."),
        ParsedAction("research", "Hamlet playwright biography"),
        ParsedAction("think", "The second result confirms Shakespeare as the playwright."),
        ParsedAction("research", "Hamlet written by who"),
        ParsedAction("think", "The third result confirms the same author."),
        ParsedAction("summary", "The retrieved documents agree that Shakespeare wrote Hamlet."),
        ParsedAction("answer", "William Shakespeare"),
    )
    for bonus, expected_score in (("0", 2.0), ("0.1", 2.1)):
        with runtime_environment(SF_EXP_TASK_MODE="search", SF_EXP_FIRST_RESEARCH_REWARD=bonus):
            state = build_state("search")
            for action in actions:
                result = await execute_protocol_action(action, state, environment_executor=fake_environment)
                assert result.accepted
            reward = compute_reward(reward_sample("search", "William Shakespeare", state))
            assert reward["score"] == expected_score
            assert reward["research_transition_reward"] == float(bonus)
            assert reward["research_completed"]
            assert state.retrieval_count == 3


# 数据：完整 Math 动作序列。算法：验证六层 credit、结果奖励和多次 action。
async def test_full_math() -> None:
    with runtime_environment(SF_EXP_TASK_MODE="math"):
        state = build_state("math")
        actions = [
            ParsedAction("skill_call", CODE_SKILL_NAME),
            ParsedAction("think", "I will compute the expression with Python and inspect the result."),
            ParsedAction("code", "value = 6 * 7\nprint(value)"),
            ParsedAction("think", "The result is 42 and I will verify it with another calculation."),
            ParsedAction("code", "check = sum([40, 2])\nprint(check)"),
            ParsedAction("think", "The second result confirms 42 and supports the answer."),
            ParsedAction("boxed_answer", "42"),
        ]
        for action in actions:
            result = await execute_protocol_action(action, state, environment_executor=fake_environment)
            assert result.accepted
        reward = compute_reward(reward_sample("math", "42", state))
        assert reward["score"] == 2.0
        assert reward["protocol_progress_reward"] == 1.0
        assert reward["outcome_reward"] == 1.0
        assert reward["action_count"] == 2


# 数据：先违规再合法 skill_call。算法：验证有 hint 时只局部乘 recovery factor。
async def test_recovery_discount() -> None:
    with runtime_environment(SF_EXP_TASK_MODE="math"):
        state = build_state("math")
        rejected = await execute_protocol_action(
            ParsedAction("think", "I reasoned before loading the required skill."),
            state,
            environment_executor=fake_environment,
        )
        assert not rejected.accepted and "<hint>" in rejected.observation
        await execute_protocol_action(
            ParsedAction("skill_call", CODE_SKILL_NAME),
            state,
            environment_executor=fake_environment,
        )
        reward = compute_reward(reward_sample("math", "42", state))
        assert reward["skill_transition_reward"] == 0.09
        assert reward["recovery_discount"] == 0.01


# 数据：Qwen 模板已打开的首轮 think。算法：跳过该非动作但保留后续真实违规的检查。
async def test_prefilled_initial_think() -> None:
    assert is_initial_prefilled_think("<|im_start|>assistant\n<think>\n", "plan text</think>")
    assert not is_initial_prefilled_think("<|im_start|>assistant\n", "plan text</think>")
    assert not is_initial_prefilled_think("<|im_start|>assistant\n<think>\n", "plan text")
    with runtime_environment(SF_EXP_TASK_MODE="search"):
        state = build_state("search")
        hint = prefilled_think_hint_observation(state)
        assert "Ready. State: no skill is loaded." in hint
        assert state.violations == [] and not state.used_recovery_hint
        for action in (
            ParsedAction("skill_call", SEARCH_SKILL_NAME),
            ParsedAction("think", "I will retrieve the author and verify the evidence."),
            ParsedAction("search", "Hamlet author"),
            ParsedAction("think", "The retrieved document identifies the author of Hamlet."),
            ParsedAction("answer", "William Shakespeare"),
        ):
            assert (await execute_protocol_action(
                action, state, environment_executor=fake_environment
            )).accepted
        sample = reward_sample("search", "William Shakespeare", state)
        sample.reward = compute_reward(sample)
        assert sample.reward["protocol_completed"]
        assert not sample.reward["used_recovery_hint"]
        assert compute_eval_metrics([sample])[0]["recovery_free_protocol_completion_rate"] == 1.0

        rejected = await execute_protocol_action(
            ParsedAction("think", "I reasoned before loading the skill."),
            build_state("search"),
            environment_executor=fake_environment,
        )
        assert not rejected.accepted and "Action rejected." in rejected.observation

    historical = {
        "prompt": "<|im_start|>assistant\n<think>\n",
        "response": "plan text</think>\n<hint>Action rejected. State: no skill is loaded. Next: load a skill.</hint>",
        "metadata": {"skill_protocol_state": {
            "used_fields": ["no_action", "skill_call"],
            "violations": ["no_action"],
        }},
    }
    assert has_prefill_rejection(historical)
    records = [
        {
            **historical,
            "dataset": "nq",
            "reward": {"protocol_completed": True, "used_recovery_hint": True},
        },
        {
            **historical,
            "dataset": "nq",
            "metadata": {"skill_protocol_state": {
                "used_fields": ["no_action", "skill_call", "think"],
                "violations": ["no_action", "premature_think"],
            }},
            "reward": {"protocol_completed": True, "used_recovery_hint": True},
        },
    ]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "samples.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        rates = audit_file(path)["total"]
    assert rates["pcr"] == 1.0
    assert rates["raw_rf_pcr"] == 0.0
    assert rates["prefill_excluded_rf_pcr"] == 0.5
    historical["response"] = historical["response"].replace("Action rejected.", "Ready.")
    assert not has_prefill_rejection(historical)


# 数据：B00 的无 hint 与无 transition credit 条件。算法：确认状态仍继续但过程分为零。
async def test_hint_and_credit_switches() -> None:
    with runtime_environment(
        SF_EXP_TASK_MODE="math",
        SF_EXP_STATE_AWARE_HINTS="0",
        SF_EXP_TRANSITION_CREDIT="0",
    ):
        state = build_state("math")
        rejected = await execute_protocol_action(
            ParsedAction("think", "This action is intentionally out of phase."),
            state,
            environment_executor=fake_environment,
        )
        assert not rejected.accepted and rejected.observation == ""
        accepted = await execute_protocol_action(
            ParsedAction("skill_call", CODE_SKILL_NAME),
            state,
            environment_executor=fake_environment,
        )
        assert accepted.accepted and "<skill_body>" in accepted.observation
        assert "<hint>" not in accepted.observation
        reward = compute_reward(reward_sample("math", "42", state))
        assert reward["protocol_progress_reward"] == 0.0
        assert reward["raw_protocol_progress_reward"] == 0.1
        assert reward["withheld_transition_credit"] == 0.1
        assert not reward["used_recovery_hint"]


# 数据：C1 的成功环境执行。算法：保留 valid transition 并撤除 observation 语义内容。
async def test_observation_withholding() -> None:
    with runtime_environment(
        SF_EXP_TASK_MODE="math",
        SF_EXP_OBSERVATION_MODE="withheld",
    ):
        state = build_state("math")
        for action in (
            ParsedAction("skill_call", CODE_SKILL_NAME),
            ParsedAction("think", "I will compute a reliable result with Python."),
            ParsedAction("code", "value = 6 * 7\nprint(value)"),
        ):
            result = await execute_protocol_action(action, state, environment_executor=fake_environment)
        assert result.accepted and "Output unavailable." in result.observation
        assert state.valid_observation_count == 1
        assert state.observation_withheld_count == 1


# 数据：论文 RQ3 的 skill 干预。算法：检查 body 内容、来源和哈希互不混淆。
def test_skill_variants() -> None:
    hashes: set[str] = set()
    for variant in ("full", "removed", "paraphrased"):
        with runtime_environment(SF_EXP_TASK_MODE="search", SF_EXP_SKILL_VARIANT=variant):
            injected = injected_skill_body(SEARCH_SKILL_NAME)
            assert len(injected.sha256) == 64
            hashes.add(injected.sha256)
            if variant == "removed":
                assert injected.body == "" and injected.source_skill_name is None
    assert len(hashes) == 3
    with runtime_environment(SF_EXP_TASK_MODE="joint", SF_EXP_SKILL_VARIANT="wrong"):
        injected = injected_skill_body(CODE_SKILL_NAME)
        assert injected.source_skill_name == SEARCH_SKILL_NAME
        assert "<search>" in injected.body


# 数据：single/joint task mode。算法：验证 system prompt 只公开相应 skill 目录。
def test_prompt_modes() -> None:
    math_prompt = build_system_prompt("math")
    search_prompt = build_system_prompt("search")
    joint_prompt = build_system_prompt("joint")
    assert CODE_SKILL_NAME in math_prompt and SEARCH_SKILL_NAME not in math_prompt
    assert SEARCH_SKILL_NAME in search_prompt and CODE_SKILL_NAME not in search_prompt
    assert CODE_SKILL_NAME in joint_prompt and SEARCH_SKILL_NAME in joint_prompt


# 数据：Qwen3.5-4B 的 Math、Search、Joint 主条件。算法：检查单模型注册表及 checkpoint 路径。
def test_registry() -> None:
    registry = load_registry()
    ids = {item["id"] for item in registry["experiments"]}
    assert registry["models"].keys() == {"qwen35_4b"}
    assert ids == {
        *(f"main_qwen35_4b_o_{task}" for task in ("math", "search", "joint")),
        "diag_qwen35_4b_o_search_first_research_reward",
    }
    for task in ("math", "search", "joint"):
        experiment_id = f"main_qwen35_4b_o_{task}"
        environment = build_environment(registry, experiment_id, phase="train", seed=20260814, root=Path.cwd())
        assert experiment_id in environment["CHECKPOINT"]
        assert environment["SF_MODEL_KEY"] == "qwen35_4b"
        assert environment["SF_EXP_FIRST_RESEARCH_REWARD"] == "0.0"
    diagnostic = build_environment(
        registry,
        "diag_qwen35_4b_o_search_first_research_reward",
        phase="train",
        seed=20260814,
        root=Path.cwd(),
    )
    assert diagnostic["SF_EXP_FIRST_RESEARCH_REWARD"] == "0.1"


# 数据：全部结构化 eval YAML。算法：调用 Slime 的正式解析器并验证数据路径与关键覆盖值。
def test_eval_configs() -> None:
    eval_root = SUITE_ROOT / "data" / "prepared" / "eval"
    config_paths = sorted(eval_root.glob("*/eval_config.yaml"))
    assert len(config_paths) >= 9
    for config_path in config_paths:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert isinstance(raw, dict) and isinstance(raw.get("eval"), dict)
        defaults = dict(raw["eval"].get("defaults") or {})
        datasets = build_eval_dataset_configs(
            SimpleNamespace(),
            ensure_dataset_list(raw["eval"].get("datasets")),
            defaults,
        )
        assert datasets
        if config_path.parent.name == "ablation_search_2wiki_hotpotqa_full":
            assert [dataset.name for dataset in datasets] == ["2wikimultihopqa", "hotpotqa"]
        for dataset in datasets:
            assert dataset.name and Path(dataset.path).is_file()
            assert dataset.input_key == "prompt"
            assert dataset.label_key == "label"
            assert dataset.metadata_key == "extra_info"
            assert dataset.n_samples_per_eval_prompt == 1
            assert dataset.max_response_len in {2048, 8192}


# 数据：全部确定性 fixtures。算法：顺序执行同步和异步边界检查。
async def run_self_test() -> None:
    test_registry()
    test_eval_configs()
    test_prompt_modes()
    test_skill_variants()
    await test_full_math()
    await test_search_summary()
    await test_first_research_reward()
    await test_recovery_discount()
    await test_prefilled_initial_think()
    await test_hint_and_credit_switches()
    await test_observation_withholding()
    print("skill_following self-test passed")


if __name__ == "__main__":
    asyncio.run(run_self_test())
