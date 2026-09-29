# LOCKED: false
"""Build matched prompt and source-method data views from frozen suite data."""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from skill_following.contracts import CODE_SKILL, SEARCH_SKILL


SEARCH_R1_INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> and it "
    "will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. If you find no further external "
    "knowledge needed, you can directly provide the answer inside <answer> and "
    "</answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
)
DATASET_FILES = {
    "math": "eval_math.parquet",
    "gsm8k": "eval_gsm8k.parquet",
    "nq": "eval_nq.parquet",
    "triviaqa": "eval_triviaqa.parquet",
    "popqa": "eval_popqa.parquet",
    "hotpotqa": "eval_hotpotqa.parquet",
    "2wikimultihopqa": "eval_2wikimultihopqa.parquet",
    "musique": "eval_musique.parquet",
    "bamboogle": "eval_bamboogle.parquet",
}
MATH_DATASETS = ("math", "gsm8k")
SEARCH_DATASETS = tuple(name for name in DATASET_FILES if name not in MATH_DATASETS)


def _messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    assert isinstance(value, list) and value
    return [dict(item) for item in value]


# 数据：统一套件的一条 prompt。算法：只抽取用户问题，移除 skill-first 后缀。
def extract_question(prompt: Any) -> str:
    messages = _messages(prompt)
    user_contents = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    assert user_contents, "prompt must contain a user message"
    content = user_contents[-1].strip()
    if "Question:" in content:
        content = content.split("Question:", 1)[1]
    for suffix in (
        "\n\nSelect and load the useful skill before reasoning or answering.",
        "\n\nSelect and load the useful skill before reasoning or answering",
    ):
        if suffix in content:
            content = content.split(suffix, 1)[0]
    question = content.strip()
    assert question
    return question


# 数据：统一 JSON label。算法：规范成任务类型与非空答案列表。
def parse_label(value: Any) -> tuple[str, list[str]]:
    payload = json.loads(value) if isinstance(value, str) else dict(value)
    task_type = str(payload.get("task_type") or "").strip().lower()
    targets = payload.get("target")
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    if isinstance(targets, str):
        targets = [targets]
    targets = [str(item).strip() for item in targets or [] if str(item).strip()]
    assert task_type in {"math", "search"} and targets
    return task_type, targets


def _metadata(row: pd.Series, condition: str, task_type: str) -> dict[str, Any]:
    metadata = row.get("extra_info")
    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict()
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.update({"baseline_condition": condition, "task_type": task_type})
    return metadata


# 数据：原始问题和任务类型。算法：构造不暴露工具或 skill 的直接回答提示。
def direct_prompt(question: str, task_type: str) -> list[dict[str, str]]:
    if task_type == "math":
        content = (
            "Solve the mathematical problem directly. Do not use external tools. "
            "End with Answer: \\boxed{...}.\n\nQuestion: " + question
        )
    else:
        content = (
            "Answer the factual question directly without external tools. Return only "
            "the short final answer inside <answer>...</answer>.\n\nQuestion: " + question
        )
    return [{"role": "user", "content": content}]


# 数据：原始问题和当前完整 skill。算法：把 skill 预加载到初始 system prompt。
def preloaded_skill_prompt(question: str, task_type: str) -> list[dict[str, str]]:
    skill = CODE_SKILL if task_type == "math" else SEARCH_SKILL
    system = (
        "The following skill is already loaded. Follow its instructions and use the "
        "available environment when requested.\n\n"
        f"Skill name: {skill.name}\nSkill instructions:\n{skill.body}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]


def search_r1_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{SEARCH_R1_INSTRUCTION} Question: {question}\n"}]


# 数据：统一评测 frame 与条件名。算法：保持题目、标签和顺序，仅替换模型可见提示。
def transform_eval_frame(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        task_type, targets = parse_label(row["label"])
        question = extract_question(row["prompt"])
        metadata = _metadata(row, condition, task_type)
        if condition == "direct_qa":
            prompt: Any = direct_prompt(question, task_type)
            label: Any = row["label"]
            label_key = "label"
        elif condition == "preloaded_skill":
            prompt = preloaded_skill_prompt(question, task_type)
            label = row["label"]
            label_key = "label"
        else:
            assert condition == "source_search_r1" and task_type == "search"
            prompt = search_r1_prompt(question)
            label = {"style": "rule", "ground_truth": {"target": targets}}
            label_key = "reward_model"
        record = {
            "data_source": row.get("data_source"),
            "prompt": prompt,
            "ability": row.get("ability"),
            label_key: label,
            "extra_info": metadata,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


# 数据：统一 Math frame。算法：转换为原始 ReTool generator 所需的纯字符串问题。
def transform_retool_frame(frame: pd.DataFrame, condition: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        task_type, targets = parse_label(row["label"])
        assert task_type == "math"
        records.append(
            {
                "data_source": row.get("data_source"),
                "prompt": extract_question(row["prompt"]),
                "ability": "math",
                "label": targets[0],
                "reward_model": {"style": "rule", "ground_truth": targets[0]},
                "metadata": _metadata(row, condition, task_type),
            }
        )
    return records


# 数据：原始 ReTool 的字符串 prompt。算法：仅改变存储表示，使带 processor 的纯文本模型能够通过 Dataset 校验。
def retool_message_prompt(prompt: Any) -> list[dict[str, str]]:
    assert isinstance(prompt, str) and prompt.strip(), "ReTool prompt must be a non-empty string"
    return [{"role": "user", "content": prompt}]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


# 数据：一份 JSONL。算法：严格读取所有非空行，供兼容视图复用原始 ReTool 题目与标签。
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"missing JSONL source: {path}"
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            assert isinstance(record, dict), f"record {line_number} must be an object"
            records.append(record)
    assert records, f"empty JSONL source: {path}"
    return records


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_eval_config(
    path: Path,
    datasets: dict[str, Path],
    *,
    label_key: str,
    metadata_key: str,
    apply_chat_template: bool,
    max_response_len: int,
    custom_rm_path: str,
) -> None:
    lines = [
        "# LOCKED: false",
        "eval:",
        "  defaults:",
        "    input_key: prompt",
        f"    label_key: {label_key}",
        f"    metadata_key: {metadata_key}",
        f"    apply_chat_template: {'true' if apply_chat_template else 'false'}",
        "    n_samples_per_eval_prompt: 1",
        "    max_prompt_len: 4096",
        f"    max_response_len: {max_response_len}",
        f"    custom_rm_path: {custom_rm_path}",
        "  datasets:",
    ]
    for name, dataset_path in datasets.items():
        lines.extend((f"    - name: {name}", f"      path: {dataset_path}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 数据：原始 ReTool JSONL 文件。算法：复制为消息列表 prompt，并保持问题、标签、顺序和评测规则完全不变。
def build_retool_processor_view(root: Path) -> dict[str, int]:
    source_root = root / "data" / "prepared" / "source_retool"
    output_root = root / "data" / "prepared" / "source_retool_messages"
    filenames = (
        "train_math_gsm8k_14973.jsonl",
        "monitor_math_gsm8k_500.jsonl",
        "eval_math.jsonl",
        "eval_gsm8k.jsonl",
    )
    counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for filename in filenames:
        source_path = source_root / filename
        records = read_jsonl(source_path)
        converted: list[dict[str, Any]] = []
        for record in records:
            converted_record = dict(record)
            converted_record["prompt"] = retool_message_prompt(record.get("prompt"))
            converted.append(converted_record)
        write_jsonl(output_root / filename, converted)
        counts[filename] = len(converted)
        source_hashes[filename] = file_sha256(source_path)

    write_eval_config(
        output_root / "full_eval_config.yaml",
        {
            "math": (output_root / "eval_math.jsonl").resolve(),
            "gsm8k": (output_root / "eval_gsm8k.jsonl").resolve(),
        },
        label_key="label",
        metadata_key="metadata",
        apply_chat_template=False,
        max_response_len=8192,
        custom_rm_path="skill_following_baselines.metrics.reward_func",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "representation": "single-user-message-list",
                "source": str(source_root.resolve()),
                "source_sha256": source_hashes,
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return counts


def build_all(root: Path, suite_root: Path) -> dict[str, int]:
    output_root = root / "data" / "prepared"
    source_eval_root = suite_root / "data" / "prepared" / "eval"
    frames: dict[str, pd.DataFrame] = {}
    for dataset_name, filename in DATASET_FILES.items():
        profile = "main_math" if dataset_name in MATH_DATASETS else "main_search"
        source = source_eval_root / profile / filename
        assert source.is_file(), f"missing source eval data: {source}"
        frames[dataset_name] = pd.read_parquet(source)

    counts: dict[str, int] = {}
    for condition in ("direct_qa", "preloaded_skill"):
        paths: dict[str, Path] = {}
        for name, frame in frames.items():
            destination = output_root / condition / f"eval_{name}.parquet"
            transformed = transform_eval_frame(frame, condition)
            destination.parent.mkdir(parents=True, exist_ok=True)
            transformed.to_parquet(destination, index=False)
            paths[name] = destination.resolve()
            counts[f"{condition}:{name}"] = len(transformed)
        write_eval_config(
            output_root / condition / "math_eval_config.yaml",
            {name: paths[name] for name in MATH_DATASETS},
            label_key="label",
            metadata_key="extra_info",
            apply_chat_template=True,
            max_response_len=8192,
            custom_rm_path="skill_following_baselines.metrics.reward_func",
        )
        write_eval_config(
            output_root / condition / "search_eval_config.yaml",
            {name: paths[name] for name in SEARCH_DATASETS},
            label_key="label",
            metadata_key="extra_info",
            apply_chat_template=True,
            max_response_len=2048,
            custom_rm_path="skill_following_baselines.metrics.reward_func",
        )

    search_paths: dict[str, Path] = {}
    for name in SEARCH_DATASETS:
        transformed = transform_eval_frame(frames[name], "source_search_r1")
        destination = output_root / "source_search_r1" / f"eval_{name}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        transformed.to_parquet(destination, index=False)
        search_paths[name] = destination.resolve()
        counts[f"source_search_r1:{name}"] = len(transformed)
    write_eval_config(
        output_root / "source_search_r1" / "full_eval_config.yaml",
        search_paths,
        label_key="reward_model",
        metadata_key="extra_info",
        apply_chat_template=True,
        max_response_len=500,
        custom_rm_path="skill_following_baselines.metrics.reward_func",
    )
    monitor_paths: dict[str, Path] = {}
    for name in ("nq", "musique"):
        monitor = pd.read_parquet(search_paths[name]).head(500)
        destination = output_root / "source_search_r1" / f"monitor_{name}_500.parquet"
        monitor.to_parquet(destination, index=False)
        monitor_paths[name] = destination.resolve()
    write_eval_config(
        output_root / "source_search_r1" / "monitor_eval_config.yaml",
        monitor_paths,
        label_key="reward_model",
        metadata_key="extra_info",
        apply_chat_template=True,
        max_response_len=500,
        custom_rm_path="generate_with_search.reward_func",
    )

    train_math_path = suite_root / "data" / "prepared" / "train" / "table1_math_14973.parquet"
    assert train_math_path.is_file()
    train_math = pd.read_parquet(train_math_path)
    retool_train = transform_retool_frame(train_math, "source_retool")
    retool_root = output_root / "source_retool"
    write_jsonl(retool_root / "train_math_gsm8k_14973.jsonl", retool_train)
    counts["source_retool:train"] = len(retool_train)
    retool_eval_paths: dict[str, Path] = {}
    for name in MATH_DATASETS:
        records = transform_retool_frame(frames[name], "source_retool")
        destination = retool_root / f"eval_{name}.jsonl"
        write_jsonl(destination, records)
        retool_eval_paths[name] = destination.resolve()
        counts[f"source_retool:{name}"] = len(records)
    monitor_records = (
        transform_retool_frame(frames["math"].head(250), "source_retool")
        + transform_retool_frame(frames["gsm8k"].head(250), "source_retool")
    )
    write_jsonl(retool_root / "monitor_math_gsm8k_500.jsonl", monitor_records)
    write_eval_config(
        retool_root / "full_eval_config.yaml",
        retool_eval_paths,
        label_key="label",
        metadata_key="metadata",
        apply_chat_template=False,
        max_response_len=8192,
        custom_rm_path="skill_following_baselines.metrics.reward_func",
    )
    processor_counts = build_retool_processor_view(root)
    counts.update({f"source_retool_messages:{name}": value for name, value in processor_counts.items()})

    manifest_path = root / "data" / "manifest.json"
    search_source = root.parent.parent.parent.parent.parent / "examples" / "search-r1" / "generate_with_search.py"
    retool_source = root.parent.parent.parent.parent.parent / "examples" / "retool" / "generate_with_retool.py"
    assert search_source.is_file() and retool_source.is_file()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "version": "skill_following_baselines",
                "counts": counts,
                "source_suite": str(suite_root.resolve()),
                "search_r1_source": {
                    "path": str(search_source.resolve()),
                    "sha256": file_sha256(search_source),
                },
                "retool_source": {
                    "path": str(retool_source.resolve()),
                    "sha256": file_sha256(retool_source),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--retool-processor-view-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.retool_processor_view_only:
        counts = build_retool_processor_view(args.root.resolve())
    else:
        assert args.suite_root is not None, "--suite-root is required for complete data preparation"
        counts = build_all(args.root.resolve(), args.suite_root.resolve())
    assert counts and all(value > 0 for value in counts.values())
    print(json.dumps({"status": "ok", "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
