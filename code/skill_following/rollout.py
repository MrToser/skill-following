# LOCKED: false
"""可干预的 Search/Code SGLang 多轮 rollout。"""

from __future__ import annotations

from typing import Any, Callable

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .config import get_runtime_config
from .contracts import (
    CODE_SKILL,
    SEARCH_SKILL,
    expected_skill_for_task,
    parse_training_label,
    skill_specs_for_mode,
)
from .prompts import USER_SKILL_INSTRUCTION
from .protocol import (
    ActionResult,
    ParsedAction,
    execute_protocol_action,
    is_initial_prefilled_think,
    parse_action,
    prefilled_think_hint_observation,
    with_protocol_stop_tags,
)
from .retool_base.rollout import append_environment_text, render_prompt_text
from .state import UnifiedSkillState
from .telemetry.trace import maybe_log_trace


DEFAULT_MAX_TURNS = 20
MAX_ASSISTANT_TURN_TOKENS = 1024


# 数据：tokenizer 与已生成 token。算法：二分搜索第一个完整 leading action 的 token 边界。
def first_complete_action_token_count(tokenizer: Any, token_ids: list[int]) -> int:
    assert token_ids
    field_name = parse_action(tokenizer.decode(token_ids)).field_name
    assert field_name is not None
    low = 1
    high = len(token_ids)
    while low < high:
        midpoint = (low + high) // 2
        if parse_action(tokenizer.decode(token_ids[:midpoint])).field_name == field_name:
            high = midpoint
        else:
            low = midpoint + 1
    assert parse_action(tokenizer.decode(token_ids[:low])).field_name == field_name
    return low


# 数据：Sample replay 字段与尾部 token 数。算法：同步裁掉所有 response 对齐数组。
def trim_response_tail(sample: Sample, tail_token_count: int) -> None:
    assert 0 <= tail_token_count <= sample.response_length
    if tail_token_count == 0:
        return
    new_response_length = sample.response_length - tail_token_count
    del sample.tokens[-tail_token_count:]
    sample.response_length = new_response_length
    if sample.loss_mask is not None:
        del sample.loss_mask[new_response_length:]
    if sample.rollout_log_probs is not None:
        del sample.rollout_log_probs[new_response_length:]
    offsets = sample.rollout_top_p_token_offsets
    top_p_ids = sample.rollout_top_p_token_ids
    if offsets is not None or top_p_ids is not None:
        assert offsets is not None and top_p_ids is not None
        new_offsets = offsets[: new_response_length + 1]
        sample.rollout_top_p_token_offsets = new_offsets
        sample.rollout_top_p_token_ids = top_p_ids[: int(new_offsets[-1])]
    if sample.rollout_routed_experts is not None:
        sample.rollout_routed_experts = sample.rollout_routed_experts[:-tail_token_count]
    sample._validate_response_metadata_lengths()


# 数据：统一 label 与 extra_info。算法：交叉检查 task_type、expected_skill 和 prompt 不泄漏约束。
def initialize_state(sample: Sample, prompt_text: str) -> UnifiedSkillState:
    runtime = get_runtime_config()
    label = parse_training_label(sample.label)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_type = str(metadata.get("task_type") or label.task_type).strip().lower()
    assert task_type == label.task_type, "metadata task_type must match label"
    expected_spec = expected_skill_for_task(task_type)
    expected_skill = str(metadata.get("expected_skill") or expected_spec.name)
    assert expected_skill == expected_spec.name
    assert runtime.accepts_task(task_type), (
        f"experiment {runtime.experiment_id} does not accept task_type={task_type}"
    )
    assert USER_SKILL_INSTRUCTION in prompt_text
    assert "Part 1: Available skills" in prompt_text
    assert "Part 2: Skill loading" in prompt_text
    available_skills = skill_specs_for_mode(runtime.task_mode)
    assert expected_skill in {skill.name for skill in available_skills}
    assert all(
        f"<skill_call>{skill.name}</skill_call>" not in prompt_text for skill in available_skills
    )
    assert CODE_SKILL.body not in prompt_text and SEARCH_SKILL.body not in prompt_text
    assert "<interpreter>" not in prompt_text and "<information>" not in prompt_text
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


# 数据：rollout args 与默认 turn 上限。算法：优先使用 custom config，并限制为正整数。
def resolve_max_turns(args: Any) -> int:
    max_turns = int(getattr(args, "max_turns", None) or DEFAULT_MAX_TURNS)
    assert max_turns > 0
    return max_turns


# 数据：slime generate 输入与 reward 预览函数。算法：执行 skill-first 状态机、首动作裁剪和环境 loss mask。
async def generate_unified(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    reward_preview_fn: Callable[[Sample], dict[str, Any]],
) -> Sample:
    assert not args.partial_rollout, "partial rollout is not supported"
    assert isinstance(sample, Sample)

    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []

    generate_state = GenerateState(args)
    prompt_text = render_prompt_text(generate_state.tokenizer, sample.prompt)
    state = initialize_state(sample, prompt_text)
    prompt_token_ids = generate_state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_token_ids)
    response = ""
    response_token_ids: list[int] = []
    sampling_params = with_protocol_stop_tags(sampling_params)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    max_context_length = (
        int(args.rollout_max_context_len)
        if args.rollout_max_context_len is not None
        else int(args.context_parallel_size) * int(args.max_tokens_per_gpu)
    )
    last_finish_reason = "stop"

    for turn_index in range(resolve_max_turns(args)):
        state.turn_index = turn_index
        total_length = len(prompt_token_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            last_finish_reason = "length"
            break
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            int(sampling_params.get("max_new_tokens", remaining_budget)),
            remaining_budget,
            MAX_ASSISTANT_TURN_TOKENS,
        )
        output = await post(
            url,
            {
                "input_ids": prompt_token_ids + response_token_ids,
                "sampling_params": per_turn_sampling_params,
                "return_logprob": True,
            },
        )
        last_finish_reason = output["meta_info"]["finish_reason"]["type"]
        if last_finish_reason == "abort" or "output_token_logprobs" not in output["meta_info"]:
            sample.status = Sample.Status.ABORTED
            return sample

        current_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        current_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        current_response = generate_state.tokenizer.decode(current_token_ids)
        prefilled_think = turn_index == 0 and is_initial_prefilled_think(
            prompt_text, current_response
        )
        parsed_action: ParsedAction = (
            ParsedAction(field_name=None, content="")
            if prefilled_think
            else parse_action(current_response)
        )
        keep_token_count = len(current_token_ids)
        if parsed_action.field_name is not None:
            keep_token_count = first_complete_action_token_count(generate_state.tokenizer, current_token_ids)

        sample.append_response_tokens(
            args,
            tokens=current_token_ids,
            log_probs=current_log_probs,
            trainable=True,
            meta_info=output["meta_info"],
            text=current_response,
        )
        if keep_token_count < len(current_token_ids):
            discarded_count = len(current_token_ids) - keep_token_count
            trim_response_tail(sample, discarded_count)
            current_token_ids = current_token_ids[:keep_token_count]
            current_response = generate_state.tokenizer.decode(current_token_ids)
            parsed_action = parse_action(current_response)
            assert parsed_action.field_name is not None
            state.record_action_tail_trim(parsed_action.field_name, discarded_count)

        response += current_response
        response_token_ids += current_token_ids
        sample.response = response
        if last_finish_reason == "length" and parsed_action.field_name is None:
            break

        action_result = (
            ActionResult(observation=prefilled_think_hint_observation(state), accepted=True)
            if prefilled_think
            else await execute_protocol_action(parsed_action, state)
        )
        if action_result.done:
            last_finish_reason = "stop"
            break
        response, response_token_ids, observation_truncated = append_environment_text(
            args=args,
            sample=sample,
            state=generate_state,
            prompt_tokens_ids=prompt_token_ids,
            response=response,
            response_token_ids=response_token_ids,
            environment_text=action_result.observation,
            max_context_length=max_context_length,
        )
        if observation_truncated:
            last_finish_reason = "length"
            break

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.prompt = prompt_text
    sample.metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata["skill_protocol_state"] = state.to_dict()
    sample.tool_call_count = state.action_count
    if state.done:
        sample.status = Sample.Status.COMPLETED
    elif last_finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.TRUNCATED

    maybe_log_trace(
        args=args,
        tokenizer=generate_state.tokenizer,
        sample=sample,
        prompt_text=prompt_text,
        response_token_ids=response_token_ids,
        state=state,
        reward_fn=reward_preview_fn,
    )
    return sample
