#!/usr/bin/env python3
# LOCKED: false
"""Prepare plain-prompt JSONL data for the original Retool baseline."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


class my_logger:
    """Small stdout logger used by released scripts."""

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def warning(self, message: str) -> None:
        print(f"[WARN] {message}")

    def warn(self, message: str) -> None:
        self.warning(message)


LOGGER = my_logger()


# Data: a JSONL path. Algorithm: stream non-empty JSON objects in order.
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.exists(), f"missing source path: {path}"
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert isinstance(record, dict), f"line {line_index} is not a JSON object"
            records.append(record)
    assert records, f"source path has no records: {path}"
    return records


# Data: prompt field from skill/base records. Algorithm: keep plain text, or
# extract the latest user message from a chat-style prompt.
def extract_user_prompt(record: dict[str, Any], prompt_key: str) -> str:
    prompt = record.get(prompt_key)
    assert prompt is not None, f"record missing prompt key: {prompt_key}"

    if isinstance(prompt, str):
        text = prompt
    else:
        assert isinstance(prompt, list), f"unsupported prompt type: {type(prompt)}"
        user_messages = [message for message in prompt if message.get("role") == "user"]
        assert user_messages, "chat prompt has no user message"
        text = user_messages[-1].get("content", "")

    assert isinstance(text, str), f"user prompt must be a string, got {type(text)}"
    text = text.strip()
    assert text, "empty user prompt"
    return text


# Data: label fields from Retool/DAPO/MATH records. Algorithm: prefer label_key,
# then fall back to reward_model.ground_truth, and normalize to string.
def extract_label(record: dict[str, Any], label_key: str) -> str:
    label = record.get(label_key)
    if label is None:
        reward_model = record.get("reward_model") or {}
        assert isinstance(reward_model, dict), "reward_model must be a dict when used as fallback"
        label = reward_model.get("ground_truth")

    assert label is not None, f"record missing label key: {label_key}"
    label_text = str(label).strip()
    assert label_text != "", "empty label"
    return label_text


# Data: one skill-format record. Algorithm: project it to the original Retool
# baseline contract: prompt is only the math question, label is the answer.
def build_base_record(
    record: dict[str, Any],
    *,
    source_path: Path,
    source_index: int,
    prompt_key: str,
    label_key: str,
) -> dict[str, Any]:
    prompt = extract_user_prompt(record, prompt_key)
    label = extract_label(record, label_key)
    extra_info = record.get("extra_info") or {}
    assert isinstance(extra_info, dict), "extra_info must be a dict when present"

    return {
        "data_source": record.get("data_source", "retool_base"),
        "prompt": prompt,
        "ability": record.get("ability", "math"),
        "label": label,
        "reward_model": record.get("reward_model", {"style": "rule", "ground_truth": label}),
        "metadata": {
            "source_path": str(source_path),
            "source_index": source_index,
            "source_dataset": extra_info.get("source_dataset", record.get("data_source", "unknown")),
            "source_prompt_format": "plain_user_prompt_for_retool_base",
            "base_from_skill_prompt": isinstance(record.get(prompt_key), list),
            "math_level": extra_info.get("math_level"),
            "math_type": extra_info.get("math_type"),
            "math_subject": extra_info.get("math_subject"),
        },
    }


# Data: records and target path. Algorithm: write JSONL atomically enough for
# repeated launch-script use on local filesystems.
def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    assert records, "no records to write"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# Data: source/output paths and keys. Algorithm: read skill/base JSONL, convert
# every record, write output JSONL and optional manifest.
def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_path)
    output_path = Path(args.output_path)
    records = read_jsonl(source_path)
    output_records = [
        build_base_record(
            record,
            source_path=source_path,
            source_index=index,
            prompt_key=args.prompt_key,
            label_key=args.label_key,
        )
        for index, record in enumerate(records)
    ]
    write_jsonl(output_records, output_path)

    manifest = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "num_records": len(output_records),
        "prompt_key": args.prompt_key,
        "label_key": args.label_key,
        "contract": "prompt is a plain math question; label is a string answer for Retool reward_func",
    }
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    LOGGER.info(f"wrote {len(output_records)} records to {output_path}")
    return manifest


# Data: built-in two-record fixture. Algorithm: verify chat and plain prompt
# conversion paths without external dependencies.
def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "source.jsonl"
        output_path = tmp_path / "output.jsonl"
        manifest_path = tmp_path / "manifest.json"
        source_records = [
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "label": "4",
                "extra_info": {"source_dataset": "self_test"},
            },
            {
                "prompt": "What is 3+5?",
                "reward_model": {"style": "rule", "ground_truth": "8"},
            },
        ]
        write_jsonl(source_records, source_path)
        args = argparse.Namespace(
            source_path=str(source_path),
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            prompt_key="prompt",
            label_key="label",
        )
        manifest = run(args)
        output_records = read_jsonl(output_path)
        assert manifest["num_records"] == 2
        assert output_records[0]["prompt"] == "What is 2+2?"
        assert output_records[0]["label"] == "4"
        assert output_records[1]["prompt"] == "What is 3+5?"
        assert output_records[1]["label"] == "8"
    LOGGER.info("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=False)
    parser.add_argument("--output-path", required=False)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    assert args.source_path, "--source-path is required"
    assert args.output_path, "--output-path is required"
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
