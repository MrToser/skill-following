# LOCKED: false
"""Run the canonical Harness from a skill that is already present in the prompt."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from skill_following.config import get_runtime_config
from skill_following.contracts import ProtocolPhase, expected_skill_for_task, parse_training_label
from skill_following.protocol import execute_protocol_action, parse_action, with_protocol_stop_tags
from skill_following.retool_base.rollout import append_environment_text, render_prompt_text
from skill_following.rollout import (
    MAX_ASSISTANT_TURN_TOKENS,
    first_complete_action_token_count,
    resolve_max_turns,
    trim_response_tail,
)
from skill_following.state import UnifiedSkillState


# 数据：预加载 prompt 与统一 label。算法：从 NEED_PLAN 开始，不要求模型重新调用 skill。
def initialize_preloaded_state(sample: Sample, prompt_text: str) -> UnifiedSkillState:
    runtime = get_runtime_config()
    label = parse_training_label(sample.label)
    spec = expected_skill_for_task(label.task_type)
    assert spec.body in prompt_text, "preloaded prompt must contain the complete current skill"
    assert runtime.accepts_task(label.task_type)
    state = UnifiedSkillState(
        expected_task_type=label.task_type,
        expected_skill=spec.name,
        experiment_id=runtime.experiment_id,
        skill_variant="full",
        state_aware_hints=runtime.state_aware_hints,
        transition_credit_enabled=False,
        recovery_factor=runtime.recovery_factor,
        observation_mode=runtime.observation_mode,
        phase=ProtocolPhase.NEED_PLAN,
        loaded_skill=spec.name,
    )
    state.record_injected_skill(spec.name, sha256(spec.body.encode("utf-8")).hexdigest())
    return state


# 数据：基础模型、预加载 skill 和当前环境。算法：复用 canonical parser、状态机与 adapter。
async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    assert not args.partial_rollout
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []

    generate_state = GenerateState(args)
    prompt_text = render_prompt_text(generate_state.tokenizer, sample.prompt)
    state = initialize_preloaded_state(sample, prompt_text)
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
        parsed_action = parse_action(current_response)
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
            trim_response_tail(sample, len(current_token_ids) - keep_token_count)
            current_token_ids = current_token_ids[:keep_token_count]
            current_response = generate_state.tokenizer.decode(current_token_ids)
            parsed_action = parse_action(current_response)
        response += current_response
        response_token_ids += current_token_ids
        sample.response = response
        if last_finish_reason == "length" and parsed_action.field_name is None:
            break
        action_result = await execute_protocol_action(parsed_action, state)
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
    sample.metadata["initial_skill_preloaded"] = True
    sample.tool_call_count = state.action_count
    if state.done:
        sample.status = Sample.Status.COMPLETED
    elif last_finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.TRUNCATED
    return sample


__all__ = ["generate", "initialize_preloaded_state"]
