# LOCKED: false
"""Prepare the complete Hendrycks MATH test split for joint evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


class my_logger:
    """Data: a module logger. Algorithm: expose a small consistent logging API."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("prepare_full_math")
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)


BOXED_START = re.compile(r"\\(?:boxed|fbox)\s*\{")


def extract_last_boxed_answer(solution: Any) -> str:
    """Data: one MATH solution. Algorithm: recover the last balanced boxed answer."""

    assert solution is not None, "solution must not be None"
    text = str(solution)
    matches = list(BOXED_START.finditer(text))
    assert matches, "MATH solution has no boxed answer"
    for match in reversed(matches):
        opening = match.end() - 1
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    answer = text[opening + 1 : index].strip()
                    assert answer, "boxed answer must not be empty"
                    return answer
    raise AssertionError("MATH solution has an unbalanced boxed answer")


def collect_records(input_root: Path, split: str, logger: my_logger) -> list[dict[str, Any]]:
    """Data: subject parquet files. Algorithm: read every subject and extract labels."""

    assert input_root.is_dir(), f"input root does not exist: {input_root}"
    files = sorted(input_root.glob(f"*/{split}-*.parquet"))
    assert files, f"no {split} parquet files under {input_root}"
    records: list[dict[str, Any]] = []
    for path in files:
        frame = pd.read_parquet(path)
        assert {"problem", "solution"}.issubset(frame.columns), f"invalid columns in {path}"
        for source_index, row in enumerate(frame.to_dict(orient="records")):
            problem = str(row["problem"]).strip()
            assert problem, f"empty problem in {path}:{source_index}"
            records.append(
                {
                    "problem": problem,
                    "answer": extract_last_boxed_answer(row["solution"]),
                    "data_source": "math",
                    "source_file": str(path),
                    "source_index": source_index,
                }
            )
        logger.info(f"loaded {path.parent.name}: {len(frame)} rows")
    assert records, "no MATH records collected"
    return records


def write_records(records: list[dict[str, Any]], output: Path, expected_rows: int | None) -> None:
    """Data: normalized MATH records. Algorithm: validate cardinality and write parquet."""

    if expected_rows is not None:
        assert len(records) == expected_rows, f"expected {expected_rows} rows, got {len(records)}"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    assert output.is_file(), f"failed to write {output}"


def parse_args() -> argparse.Namespace:
    """Data: command-line arguments. Algorithm: define the bounded preparation interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    """Data: CLI configuration. Algorithm: collect, validate, persist, and summarize."""

    args = parse_args()
    logger = my_logger()
    records = collect_records(args.input_root, args.split, logger)
    write_records(records, args.output, args.expected_rows)
    manifest = {
        "source": str(args.input_root),
        "split": args.split,
        "rows": len(records),
        "output": str(args.output),
        "subjects": sorted({Path(row["source_file"]).parent.name for row in records}),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"wrote complete MATH evaluation source: {args.output} ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
