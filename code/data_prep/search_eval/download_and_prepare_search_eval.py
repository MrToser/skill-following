#!/usr/bin/env python3
# LOCKED: false
"""Download missing Search-R1 evaluation files and build a full eval view.

The existing NQ, HotpotQA, and Musique files from experiment 01 are reused.
Only TriviaQA, PopQA, 2WikiMultihopQA, and Bamboogle are downloaded from the
same FlashRAG dataset repository used by Search-R1. Raw files are retained,
and the processed output is written to a dedicated directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(os.environ.get("ROOT_PATH", Path.cwd()))
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixed_data.prompts import build_prompt_messages  # noqa: E402


REPO_URL = "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main"
DOWNLOAD_SPECS: dict[str, tuple[str, str]] = {
    "triviaqa": ("test", "triviaqa/test.jsonl"),
    "popqa": ("test", "popqa/test.jsonl"),
    "2wikimultihopqa": ("dev", "2wikimultihopqa/dev.jsonl"),
    "bamboogle": ("test", "bamboogle/test.jsonl"),
}
EXISTING_DATASETS = ("nq", "hotpotqa", "musique")
ALL_DATASETS = EXISTING_DATASETS + tuple(DOWNLOAD_SPECS)


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def normalize_question(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        raise ValueError("empty question")
    return text


def extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "problem"):
        if isinstance(row.get(key), str) and row[key].strip():
            return normalize_question(row[key])
    prompt = json_safe(row.get("prompt"))
    if isinstance(prompt, list):
        user_messages = [
            str(item.get("content", ""))
            for item in prompt
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        prompt = user_messages[-1] if user_messages else ""
    text = str(prompt or "")
    if "Question:" in text:
        text = text.rsplit("Question:", 1)[1]
    text = text.split("\n\nSelect and load the useful skill", 1)[0]
    return normalize_question(text)


def _target_list(value: Any) -> list[str]:
    value = json_safe(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate[:1] in "[{":
            try:
                return _target_list(json.loads(candidate))
            except json.JSONDecodeError:
                pass
        return [candidate] if candidate else []
    if isinstance(value, dict):
        for key in ("target", "answer", "golden_answers"):
            if key in value:
                return _target_list(value[key])
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def extract_targets(row: dict[str, Any]) -> list[str]:
    candidates = [row.get("golden_answers"), row.get("answer"), row.get("label")]
    reward_model = json_safe(row.get("reward_model"))
    if isinstance(reward_model, dict):
        candidates.extend([reward_model.get("ground_truth"), reward_model.get("target")])
    for candidate in candidates:
        targets = _target_list(candidate)
        if targets:
            return targets
    raise ValueError(f"no answer targets in row keys={sorted(row)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "skill-following-data-prep/0.002"})
            with urlopen(request, timeout=90) as response, partial.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            os.replace(partial, destination)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def existing_rows(path: Path) -> Iterable[dict[str, Any]]:
    frame = pd.read_parquet(path)
    for index, raw in enumerate(frame.to_dict(orient="records")):
        row = json_safe(raw)
        dataset = str(row.get("data_source", "")).strip().lower()
        if dataset not in EXISTING_DATASETS:
            continue
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        yield {
            "data_source": dataset,
            "question": extract_question(row),
            "golden_answers": extract_targets(row),
            "source_kind": "processed_01",
            "source_path": str(path),
            "source_index": index,
            "source_split": str(extra.get("split", "test")),
        }


def downloaded_rows(raw_dir: Path) -> Iterable[dict[str, Any]]:
    for dataset, (split, relative_path) in DOWNLOAD_SPECS.items():
        path = raw_dir / relative_path
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            yield {
                "data_source": dataset,
                "question": extract_question(row),
                "golden_answers": extract_targets(row),
                "source_kind": "downloaded_hf",
                "source_path": str(path),
                "source_index": index,
                "source_split": split,
            }


def build_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row["data_source"]
    targets = [str(target).strip() for target in row["golden_answers"] if str(target).strip()]
    if not targets:
        raise ValueError(f"empty targets for {dataset}:{row['source_index']}")
    return {
        "data_source": dataset,
        "prompt": build_prompt_messages(row["question"]),
        "ability": "fact-reasoning",
        "label": json.dumps(
            {"task_type": "search", "target": targets},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "extra_info": {
            "task_type": "search",
            "expected_skill": "qa-search-protocol",
            "source_dataset": dataset,
            "source": row["source_path"],
            "source_index": row["source_index"],
            "source_split": row["source_split"],
            "split": "eval",
            "source_kind": row["source_kind"],
            "local_only": row["source_kind"] == "processed_01",
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/02_prepared_data/search_eval_full",
    )
    parser.add_argument(
        "--existing-search",
        type=Path,
        default=ROOT / "data/02_prepared_data/local_eval/search_local/test.parquet",
    )
    parser.add_argument("--download", action="store_true", help="download missing raw files when absent")
    parser.add_argument("--limit-per-dataset", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit_per_dataset <= 0:
        raise ValueError("--limit-per-dataset must be positive")
    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "00_metadata"
    raw_dir = output_dir / "01_sources" / "downloaded_flashrag"
    intermediate_dir = output_dir / "02_intermediate"
    processed_root = output_dir / "03_processed"
    full_dir = processed_root / "full"
    eval_dir = processed_root / f"eval_{args.limit_per_dataset}"
    for directory in (metadata_dir, raw_dir, intermediate_dir, full_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_manifest: dict[str, Any] = {}
    for dataset, (_, relative_path) in DOWNLOAD_SPECS.items():
        path = raw_dir / relative_path
        if args.download:
            download_file(f"{REPO_URL}/{relative_path}", path)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        raw_manifest[dataset] = {
            "split": DOWNLOAD_SPECS[dataset][0],
            "path": str(path),
            "rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "url": f"{REPO_URL}/{relative_path}",
        }

    rows = list(existing_rows(args.existing_search)) + list(downloaded_rows(raw_dir))
    if not rows:
        raise RuntimeError("no Search evaluation rows found")
    counts = {dataset: sum(row["data_source"] == dataset for row in rows) for dataset in ALL_DATASETS}
    combined_path = intermediate_dir / "combined_search_source.jsonl"
    write_jsonl(rows, combined_path)

    eval_counts: dict[str, int] = {}
    full_counts: dict[str, int] = {}
    merged_full_rows: list[dict[str, Any]] = []
    merged_eval_rows: list[dict[str, Any]] = []
    for dataset in ALL_DATASETS:
        dataset_rows = [row for row in rows if row["data_source"] == dataset]
        if not dataset_rows:
            raise RuntimeError(f"missing required dataset after preparation: {dataset}")
        full_eval_rows = [build_eval_row(row) for row in dataset_rows]
        capped_rows = full_eval_rows[: args.limit_per_dataset]
        merged_full_rows.extend(full_eval_rows)
        merged_eval_rows.extend(capped_rows)
        pd.DataFrame(full_eval_rows).to_parquet(full_dir / f"eval_{dataset}.parquet", index=False)
        pd.DataFrame(capped_rows).to_parquet(eval_dir / f"eval_{dataset}.parquet", index=False)
        full_counts[dataset] = len(full_eval_rows)
        eval_counts[dataset] = len(capped_rows)
    pd.DataFrame(merged_full_rows).to_parquet(full_dir / "test.parquet", index=False)
    pd.DataFrame(merged_eval_rows).to_parquet(eval_dir / "test.parquet", index=False)

    manifest = {
        "producer": "search_eval.download_and_prepare_search_eval",
        "dataset_repo": "RUC-NLPIR/FlashRAG_datasets",
        "downloaded_datasets": raw_manifest,
        "reused_datasets": {
            dataset: str(args.existing_search) for dataset in EXISTING_DATASETS
        },
        "available_datasets": list(ALL_DATASETS),
        "missing_datasets": [],
        "combined_source": str(combined_path),
        "metadata_dir": str(metadata_dir),
        "full_processed_dir": str(full_dir),
        "eval_dir": str(eval_dir),
        "full_merged_path": str(full_dir / "test.parquet"),
        "merged_eval_path": str(eval_dir / "test.parquet"),
        "full_row_counts": full_counts,
        "eval_row_counts": eval_counts,
        "source_row_counts": counts,
        "limit_per_dataset": args.limit_per_dataset,
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
