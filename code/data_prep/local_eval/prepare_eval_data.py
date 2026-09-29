#!/usr/bin/env python3
# LOCKED: false
"""Convert already processed local Math/Search/GSM8K records to one eval schema.

This module reads the files supplied by the caller and records absent requested
datasets in the manifest. It never downloads data itself. ``--source-origin``
distinguishes the original local 01 view from an explicitly downloaded view.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mixed_data.prompts import build_prompt_messages


REQUESTED_SEARCH = (
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
)


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        return [json_safe(row) for row in frame.to_dict(orient="records")]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"non-object row in {path}")
                rows.append(row)
    return rows


def question_from_row(row: dict[str, Any]) -> str:
    for key in ("question", "problem"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    prompt = json_safe(row.get("prompt"))
    if isinstance(prompt, list):
        contents = [
            str(item.get("content", ""))
            for item in prompt
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        prompt = contents[-1] if contents else ""
    text = str(prompt or "")
    if "Question:" in text:
        text = text.rsplit("Question:", 1)[1]
    text = text.split("\n\nSelect and load the useful skill", 1)[0]
    text = text.split("\n\n", 1)[0] if text.startswith("Answer the given question") else text
    normalized = " ".join(text.strip().split())
    if not normalized:
        raise ValueError(f"cannot recover question from row keys={sorted(row)}")
    return normalized


def as_targets(value: Any) -> list[str]:
    value = json_safe(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate[:1] in "[{":
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return as_targets(parsed)
    if isinstance(value, dict):
        value = value.get("target", value.get("ground_truth"))
        if isinstance(value, dict):
            value = value.get("target", value.get("answer", value))
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    targets = [str(item).strip() for item in values if str(item).strip()]
    return targets


def targets_from_row(row: dict[str, Any], *, gsm8k: bool = False) -> list[str]:
    candidates = [row.get("label"), row.get("answer"), row.get("golden_answers")]
    reward_model = json_safe(row.get("reward_model"))
    if isinstance(reward_model, dict):
        candidates.extend([reward_model.get("ground_truth"), reward_model.get("target")])
    for candidate in candidates:
        targets = as_targets(candidate)
        if targets:
            if gsm8k:
                normalized: list[str] = []
                for target in targets:
                    match = re.search(r"####\s*(.+?)\s*$", target, flags=re.DOTALL)
                    normalized.append((match.group(1) if match else target).strip())
                targets = normalized
            return targets
    raise ValueError(f"cannot recover answer target from row keys={sorted(row)}")


def build_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    source_path: Path,
    source_index: int,
    task_type: str,
    gsm8k: bool = False,
) -> dict[str, Any]:
    targets = targets_from_row(row, gsm8k=gsm8k)
    question = question_from_row(row)
    return {
        "data_source": dataset_name,
        "prompt": build_prompt_messages(question),
        "ability": "math" if task_type == "math" else "fact-reasoning",
        "label": json.dumps(
            {"task_type": task_type, "target": targets},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "extra_info": {
            "task_type": task_type,
            "expected_skill": (
                "code-interpreter-protocol" if task_type == "math" else "qa-search-protocol"
            ),
            "source_dataset": dataset_name,
            "source": str(source_path),
            "source_index": source_index,
            "split": "eval",
            "local_only": True,
        },
    }


def write_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--math-path", type=Path, required=True)
    parser.add_argument("--search-path", type=Path, required=True)
    parser.add_argument("--gsm8k-path", type=Path)
    parser.add_argument("--limit-per-dataset", type=int, default=500)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--source-origin",
        choices=("local", "downloaded"),
        default="local",
        help="provenance label for the supplied Search/GSM8K files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit_per_dataset <= 0:
        raise ValueError("--limit-per-dataset must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = {
        "math": read_rows(args.math_path),
        "search": read_rows(args.search_path),
    }
    search_counts = Counter(str(row.get("data_source", "")).strip().lower() for row in source_rows["search"])
    available: list[str] = []
    missing: list[str] = []
    row_counts: dict[str, int] = {}

    def select(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
        selected = [
            (index, row)
            for index, row in enumerate(rows)
            if str(row.get("data_source", "")).strip().lower() == dataset_name
        ]
        selected = selected[: args.limit_per_dataset]
        return [
            build_row(
                row,
                dataset_name=dataset_name,
                source_path=args.search_path,
                source_index=index,
                task_type="search",
            )
            for index, row in selected
        ]

    math_rows = [
        build_row(
            row,
            dataset_name="math",
            source_path=args.math_path,
            source_index=index,
            task_type="math",
        )
        for index, row in enumerate(source_rows["math"][: args.limit_per_dataset])
    ]
    write_dataset(math_rows, args.output_dir / "eval_math.parquet")
    available.append("math")
    row_counts["math"] = len(math_rows)

    for dataset_name in REQUESTED_SEARCH:
        rows = select(source_rows["search"], dataset_name)
        if not rows:
            missing.append(dataset_name)
            continue
        write_dataset(rows, args.output_dir / f"eval_{dataset_name}.parquet")
        available.append(dataset_name)
        row_counts[dataset_name] = len(rows)

    if args.gsm8k_path and args.gsm8k_path.exists():
        gsm_rows = [
            build_row(
                row,
                dataset_name="gsm8k",
                source_path=args.gsm8k_path,
                source_index=index,
                task_type="math",
                gsm8k=True,
            )
            for index, row in enumerate(read_rows(args.gsm8k_path)[: args.limit_per_dataset])
        ]
        if gsm_rows:
            write_dataset(gsm_rows, args.output_dir / "eval_gsm8k.parquet")
            available.append("gsm8k")
            row_counts["gsm8k"] = len(gsm_rows)
    else:
        missing.append("gsm8k")

    if args.require_all and missing:
        raise RuntimeError(f"requested datasets missing from local inputs: {missing}")

    manifest = {
        "producer": "local_eval.prepare_eval_data",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_only": args.source_origin == "local",
        "source_origin": args.source_origin,
        "requested_search_datasets": list(REQUESTED_SEARCH),
        "available_datasets": available,
        "missing_datasets": missing,
        "row_counts": row_counts,
        "source_paths": {
            "math": str(args.math_path),
            "search": str(args.search_path),
            "gsm8k": str(args.gsm8k_path) if args.gsm8k_path else None,
        },
        "search_source_counts": dict(search_counts),
        "limit_per_dataset": args.limit_per_dataset,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
