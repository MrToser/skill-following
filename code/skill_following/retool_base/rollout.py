# LOCKED: false
"""SGLang-backed Retool rollout orchestration."""

from __future__ import annotations

from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .constants import SKILL_NAME
from .models import RetoolRuntimeConfig, RetoolSkillState
from .protocol import execute_protocol_action, parse_action, with_protocol_stop_tags
from .trace import maybe_log_training_token_trace


# Data: tokenizer and chat-style prompt. Algorithm: render Qwen chat text without tool schema leakage.
def render_prompt_text(tokenizer: Any, prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, str):
        assert prompt.strip(), "prompt string must not be empty"
        return prompt
    assert isinstance(prompt, list) and prompt, "prompt must be a string or non-empty message list"
    try:
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = []
        for message in prompt:
            role = message.get("role")
            content = message.get("content", "")
            assert role in {"system", "user", "assistant"}, f"unsupported message role: {role}"
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        rendered.append("<|im_start|>assistant\n")
        return "\n".join(rendered)


# Data: args and version-owned fallback. Algorithm: resolve a positive Retool turn limit.
def resolve_max_turns(args: Any, default_max_turns: int) -> int:
    assert default_max_turns > 0, "default_max_turns must be positive"
    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        max_turns = default_max_turns
    max_turns = int(max_turns)
    assert max_turns > 0, "max_turns must be positive"
    return max_turns


# Data: response state and environment text. Algorithm: append observation tokens with loss_mask=0 and fit context budget.
def append_environment_text(
    *,
    args: Any,
    sample: Sample,
    state: GenerateState,
    prompt_tokens_ids: list[int],
    response: str,
    response_token_ids: list[int],
    environment_text: str,
    max_context_length: int,
) -> tuple[str, list[int], bool]:
    if not environment_text:
        return response, response_token_ids, False
    obs_token_ids = state.tokenizer(environment_text, add_special_tokens=False)["input_ids"]
    overflow = len(prompt_tokens_ids) + len(response_token_ids) + len(obs_token_ids) - max_context_length
    truncated = overflow > 0
    if truncated:
        obs_token_ids = obs_token_ids[: max(0, len(obs_token_ids) - overflow)]
        environment_text = state.tokenizer.decode(obs_token_ids) if obs_token_ids else ""
    if obs_token_ids:
        response += environment_text
        response_token_ids += obs_token_ids
        sample.append_response_tokens(args, tokens=obs_token_ids, trainable=False, text=environment_text)
    return response, response_token_ids, truncated


# Data: slime custom-generate inputs plus immutable runtime policy.
# Algorithm: dynamic skill loader -> Retool sandbox loop -> version-selected hints/trace -> masked environment tokens.
async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    runtime: RetoolRuntimeConfig,
) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function."
    assert isinstance(sample, Sample), "sample must be a slime Sample"

    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    prompt_text = render_prompt_text(state.tokenizer, sample.prompt)
    assert f"<skill_call>{SKILL_NAME}</skill_call>" not in prompt_text, "initial prompt must not expose executable skill_call"
    assert "<tool_call>" not in prompt_text, "initial skill prompt must not expose tool schema"
    assert "<interpreter>" not in prompt_text, "initial skill prompt must not expose interpreter tag"
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_tokens_ids)

    response = ""
    response_token_ids: list[int] = []
    skill_state = RetoolSkillState()
    sampling_params = with_protocol_stop_tags(sampling_params)

    if args.rollout_max_context_len is not None:
        max_context_length = int(args.rollout_max_context_len)
    else:
        max_context_length = int(args.context_parallel_size) * int(args.max_tokens_per_gpu)

    last_finish_reason = "stop"
    for turn_index in range(resolve_max_turns(args, runtime.default_max_turns)):
        skill_state.turn_index = turn_index
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            int(sampling_params.get("max_new_tokens", remaining_budget)),
            remaining_budget,
        )
        payload = {
            "input_ids": prompt_tokens_ids + response_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)
        last_finish_reason = output["meta_info"]["finish_reason"]["type"]
        if last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        if "output_token_logprobs" not in output["meta_info"]:
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response = state.tokenizer.decode(cur_response_token_ids)

        response += cur_response
        response_token_ids += cur_response_token_ids
        sample.append_response_tokens(
            args,
            tokens=cur_response_token_ids,
            log_probs=cur_log_probs,
            trainable=True,
            meta_info=output["meta_info"],
            text=cur_response,
        )

        if last_finish_reason == "length":
            break

        action_result = await execute_protocol_action(parse_action(cur_response), skill_state)
        if action_result.done:
            break

        next_obs = action_result.observation
        if action_result.violation is not None:
            feedback = runtime.handle_violation(action_result.violation)
            if feedback.require_think:
                skill_state.require_think("violation")
            next_obs = feedback.observation

        response, response_token_ids, truncated_by_observation = append_environment_text(
            args=args,
            sample=sample,
            state=state,
            prompt_tokens_ids=prompt_tokens_ids,
            response=response,
            response_token_ids=response_token_ids,
            environment_text=next_obs,
            max_context_length=max_context_length,
        )
        if truncated_by_observation:
            last_finish_reason = "length"
            break

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.prompt = prompt_text
    sample.metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata["skill_protocol_state"] = skill_state.to_dict()
    sample.tool_call_count = skill_state.tool_call_count

    if last_finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif skill_state.done:
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    maybe_log_training_token_trace(
        args=args,
        tokenizer=state.tokenizer,
        sample=sample,
        prompt_text=prompt_text,
        response_token_ids=response_token_ids,
        loss_mask=sample.loss_mask,
        skill_state=skill_state,
        runtime=runtime,
    )
    return sample
