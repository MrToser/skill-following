# LOCKED: false
"""Audit historical RF-PCR with the Qwen template's first open think excluded."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..protocol import is_initial_prefilled_think


# 数据：一条已保存评测轨迹。算法：同时核对模板前缀、首个拒绝提示和状态中的首次 no_action。
def has_prefill_rejection(record: dict[str, Any]) -> bool:
    response = str(record.get("response") or "")
    first_think, closing_tag, following_text = response.partition("</think>")
    state = (record.get("metadata") or {}).get("skill_protocol_state") or {}
    return bool(
        closing_tag
        and is_initial_prefilled_think(str(record.get("prompt") or ""), first_think + closing_tag)
        and following_text.lstrip().startswith(
            "<hint>Action rejected. State: no skill is loaded."
        )
        and (state.get("used_fields") or [None])[0] == "no_action"
        and (state.get("violations") or [None])[0] == "no_action"
    )


# 数据：历史 JSONL 的逐条 reward 与状态。算法：流式统计原始及仅排除模板首拒绝的 RF-PCR。
def audit_file(path: Path) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            reward = record["reward"]
            state = record["metadata"]["skill_protocol_state"]
            counts_for_dataset = counts[str(record["dataset"])]
            completed = bool(reward["protocol_completed"])
            recovery = bool(reward["used_recovery_hint"])
            prefill_rejection = has_prefill_rejection(record)
            remaining_violations = (state.get("violations") or [])[int(prefill_rejection):]
            counts_for_dataset["samples"] += 1
            counts_for_dataset["pcr"] += int(completed)
            counts_for_dataset["raw_rf_pcr"] += int(completed and not recovery)
            counts_for_dataset["prefill_rejections"] += int(prefill_rejection)
            counts_for_dataset["adjusted_rf_pcr"] += int(
                completed and (not recovery or prefill_rejection and not remaining_violations)
            )

    datasets = {}
    total: Counter[str] = Counter()
    for name, dataset_counts in sorted(counts.items()):
        total.update(dataset_counts)
        datasets[name] = _format_counts(dataset_counts)
    return {
        "input": str(path),
        "method": "Exclude only the first no_action caused by a template-opened <think>; retain later violations. Historical feedback and rewards are unchanged.",
        "total": _format_counts(total),
        "datasets": datasets,
    }


# 数据：某数据集或全量样本的计数。算法：计算以全部样本为分母的指标比例。
def _format_counts(counts: Counter[str]) -> dict[str, int | float]:
    sample_count = counts["samples"]
    assert sample_count > 0
    return {
        "samples": sample_count,
        "prefill_rejections": counts["prefill_rejections"],
        "pcr": counts["pcr"] / sample_count,
        "raw_rf_pcr": counts["raw_rf_pcr"] / sample_count,
        "prefill_excluded_rf_pcr": counts["adjusted_rf_pcr"] / sample_count,
    }


# 数据：输入轨迹和可选输出路径。算法：只输出独立审计报告，不修改历史评测文件。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="saved eval/samples.jsonl")
    parser.add_argument("--output", type=Path, help="optional independent JSON report")
    args = parser.parse_args()
    report = json.dumps(audit_file(args.samples), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
