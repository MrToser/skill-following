# LOCKED: false
"""Aggregate Qwen3.5-4B evaluation summaries into a dataset-level CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ..logger import my_logger
from ..registry import SUITE_ROOT, load_registry, resolve_experiment


LOGGER = my_logger("skill_following.analysis")
RESULT_ROOT = SUITE_ROOT / "data" / "results"


# 数据：三个主条件的评测 summary。算法：逐数据集展开并记录缺失条件。
def analyze_main_results(*, allow_missing: bool) -> list[dict[str, Any]]:
    registry = load_registry()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in registry["experiments"]:
        experiment = resolve_experiment(registry, raw["id"])
        for seed in experiment["seeds"]:
            summary_path = (
                SUITE_ROOT / "data" / "artifacts" / experiment["id"]
                / f"seed_{seed}" / "eval" / "summary.json"
            )
            if not summary_path.is_file():
                missing.append(f"{experiment['id']}:seed={seed}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert isinstance(summary, dict)
            for dataset_name, metrics in summary.get("datasets", {}).items():
                assert isinstance(metrics, dict)
                rows.append({
                    "experiment_id": experiment["id"],
                    "model": experiment["model"],
                    "task_mode": experiment["task_mode"],
                    "seed": seed,
                    "dataset": dataset_name,
                    **metrics,
                })
    if missing and not allow_missing:
        raise FileNotFoundError(f"missing evaluation summaries: {missing}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with (RESULT_ROOT / "main_results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_ROOT / "main_results_missing.json").write_text(
        json.dumps({"missing": missing}, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("main results rows=%s missing=%s", len(rows), len(missing))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    analyze_main_results(allow_missing=args.allow_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
