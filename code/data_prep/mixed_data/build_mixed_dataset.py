# LOCKED: false
"""构建 Math、NQ、HotpotQA 等比例统一训练集。"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .contracts import (
    CODE_SKILL,
    SEARCH_SKILL,
    SYSTEM_PROMPT_VERSION,
    TaskType,
)
from .logger import my_logger
from .prompts import build_prompt_messages


LOGGER = my_logger("unified_search_code.dataset")
DEFAULT_SEED = 20260811
DEFAULT_EVAL_LIMIT = 500
DEFAULT_CHUNK_SIZE = 4096
SOURCE_ORDER = ("math", "nq", "hotpotqa")


# 数据：numpy、Arrow 或普通 Python 值。算法：递归转换为 JSON 安全类型。
def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# 数据：JSONL 文件路径。算法：逐行解析并拒绝空数据或非对象记录。
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.exists(), f"missing JSONL source: {path}"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            assert isinstance(row, dict), f"row {line_number} is not an object"
            rows.append(row)
    assert rows, f"empty JSONL source: {path}"
    return rows


# 数据：Parquet 文件路径。算法：使用 pandas 恢复嵌套记录并转成 JSON 安全字典。
def read_parquet(path: Path) -> list[dict[str, Any]]:
    assert path.exists(), f"missing Parquet source: {path}"
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required for mixed data building") from exc
    frame = pd.read_parquet(path)
    assert not frame.empty, f"empty Parquet source: {path}"
    return [json_safe(row) for row in frame.to_dict(orient="records")]


# 数据：字符串或 message list prompt。算法：提取最后一个 user message 并去掉旧 Question 前缀。
def extract_question(prompt: Any) -> str:
    prompt = json_safe(prompt)
    if isinstance(prompt, str):
        text = prompt
    else:
        assert isinstance(prompt, list) and prompt, "prompt must be text or message list"
        user_contents = [
            str(message.get("content", ""))
            for message in prompt
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        assert user_contents, "prompt has no user message"
        text = user_contents[-1]
    marker = "Question:"
    marker_index = text.rfind(marker)
    if marker_index >= 0:
        text = text[marker_index + len(marker) :]
    instruction_index = text.find("\n\nSelect and load the useful skill")
    if instruction_index >= 0:
        text = text[:instruction_index]
    normalized = " ".join(text.strip().split())
    assert normalized, "extracted question is empty"
    return normalized


# 数据：任意 ground-truth 表示。算法：规范成非空字符串列表。
def normalize_targets(value: Any) -> list[str]:
    value = json_safe(value)
    if isinstance(value, str):
        value = [value]
    assert isinstance(value, list), "ground truth must be a string or list"
    targets = [str(item).strip() for item in value if str(item).strip()]
    assert targets, "ground truth targets must not be empty"
    return targets


# 数据：Math 原始记录。算法：优先读取 label，回退 reward_model.ground_truth。
def extract_math_targets(row: dict[str, Any]) -> list[str]:
    if row.get("label") is not None:
        return normalize_targets(row["label"])
    reward_model = json_safe(row.get("reward_model"))
    assert isinstance(reward_model, dict)
    return normalize_targets(reward_model.get("ground_truth"))


# 数据：Search 原始记录。算法：读取 reward_model.ground_truth.target。
def extract_search_targets(row: dict[str, Any]) -> list[str]:
    reward_model = json_safe(row.get("reward_model"))
    assert isinstance(reward_model, dict)
    ground_truth = reward_model.get("ground_truth")
    assert isinstance(ground_truth, dict)
    return normalize_targets(ground_truth.get("target"))


# 数据：来源记录、任务信息和采样位置。算法：转换为统一 prompt、label 与扁平 metadata contract。
def build_unified_row(
    row: dict[str, Any],
    *,
    source_name: str,
    source_path: str,
    source_index: int,
    split: str,
    mix_occurrence: int,
) -> dict[str, Any]:
    assert source_name in SOURCE_ORDER
    assert source_index >= 0 and mix_occurrence >= 0
    task_type = TaskType.MATH if source_name == "math" else TaskType.SEARCH
    skill = CODE_SKILL if task_type == TaskType.MATH else SEARCH_SKILL
    targets = extract_math_targets(row) if task_type == TaskType.MATH else extract_search_targets(row)
    question = extract_question(row.get("prompt"))
    label = json.dumps(
        {"task_type": task_type.value, "target": targets},
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "data_source": source_name,
        "prompt": build_prompt_messages(question),
        "ability": "math" if task_type == TaskType.MATH else "fact-reasoning",
        "label": label,
        "extra_info": {
            "task_type": task_type.value,
            "skill_candidates": [CODE_SKILL.name, SEARCH_SKILL.name],
            "expected_skill": skill.name,
            "skill_spec_id": f"{skill.name}@direct_code",
            "environment_adapter": skill.environment_adapter,
            "reward_source": skill.reward_source,
            "answer": json.dumps(targets, ensure_ascii=False),
            "split": split,
            "source": source_path,
            "source_dataset": source_name,
            "source_index": source_index,
            "mix_occurrence": mix_occurrence,
            "prompt_version": SYSTEM_PROMPT_VERSION,
        },
    }


# 数据：单来源行数、目标行数和随机种子。算法：每轮先全量洗牌，再循环补足到目标规模。
def balanced_indices(row_count: int, target_count: int, seed: int) -> list[tuple[int, int]]:
    assert row_count > 0 and target_count >= row_count
    random_generator = random.Random(seed)
    selected: list[tuple[int, int]] = []
    occurrence = 0
    while len(selected) < target_count:
        indices = list(range(row_count))
        random_generator.shuffle(indices)
        remaining = target_count - len(selected)
        selected.extend((index, occurrence) for index in indices[:remaining])
        occurrence += 1
    assert len(selected) == target_count
    return selected


# 数据：三来源记录和混合策略。算法：构造全量或等比例 schedule 后做一次全局确定性洗牌。
def build_train_schedule(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    balance_strategy: str,
    seed: int,
) -> list[tuple[str, int, int]]:
    assert tuple(source_rows) == SOURCE_ORDER
    assert balance_strategy in {"equal", "natural"}
    target_count = max(len(rows) for rows in source_rows.values())
    schedule: list[tuple[str, int, int]] = []
    for source_offset, source_name in enumerate(SOURCE_ORDER):
        rows = source_rows[source_name]
        source_target = target_count if balance_strategy == "equal" else len(rows)
        if balance_strategy == "equal":
            indices = balanced_indices(len(rows), source_target, seed + source_offset + 1)
        else:
            indices = [(index, 0) for index in range(len(rows))]
        schedule.extend((source_name, index, occurrence) for index, occurrence in indices)
    random.Random(seed).shuffle(schedule)
    assert schedule
    return schedule


# 数据：统一 row iterable、目标路径和预计行数。算法：分块写 zstd Parquet，避免大集合二次驻留内存。
def write_parquet_rows(
    rows: Iterable[dict[str, Any]],
    path: Path,
    *,
    expected_count: int,
    chunk_size: int,
    force: bool,
) -> int:
    assert expected_count > 0 and chunk_size > 0
    if path.exists() and not force:
        raise FileExistsError(f"refuse to overwrite existing output: {path}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for mixed data building") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    schema = None
    buffer: list[dict[str, Any]] = []
    written = 0
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) < chunk_size:
                continue
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema, compression="zstd")
            writer.write_table(table)
            written += len(buffer)
            buffer = []
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema, compression="zstd")
            writer.write_table(table)
            written += len(buffer)
    finally:
        if writer is not None:
            writer.close()
    assert written == expected_count, f"expected {expected_count} rows, wrote {written}"
    return written


# 数据：训练 schedule 与来源记录。算法：按 schedule 延迟转换，供分块 writer 消费。
def iter_train_rows(
    schedule: list[tuple[str, int, int]],
    source_rows: dict[str, list[dict[str, Any]]],
    source_paths: dict[str, str],
) -> Iterator[dict[str, Any]]:
    for source_name, source_index, occurrence in schedule:
        yield build_unified_row(
            source_rows[source_name][source_index],
            source_name=source_name,
            source_path=source_paths[source_name],
            source_index=source_index,
            split="train",
            mix_occurrence=occurrence,
        )


# 数据：评估记录、上限与种子。算法：确定性抽样并转换为统一 contract。
def build_eval_rows(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
    source_path: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    assert rows and limit > 0
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected_indices = sorted(indices[: min(limit, len(indices))])
    return [
        build_unified_row(
            rows[index],
            source_name=source_name,
            source_path=source_path,
            source_index=index,
            split="eval",
            mix_occurrence=0,
        )
        for index in selected_indices
    ]


# 数据：合成 Math/Search rows。算法：检查 prompt 不泄漏、label 同构和三任务 schedule 平衡。
def run_self_test() -> None:
    math_row = {
        "prompt": [{"role": "user", "content": "What is 6 times 7?"}],
        "label": "42",
    }
    search_row = {
        "prompt": [{"role": "user", "content": "Question: Who wrote Hamlet?"}],
        "reward_model": {"ground_truth": {"target": ["William Shakespeare"]}},
    }
    built_math = build_unified_row(
        math_row,
        source_name="math",
        source_path="math.jsonl",
        source_index=0,
        split="train",
        mix_occurrence=0,
    )
    built_search = build_unified_row(
        search_row,
        source_name="nq",
        source_path="nq.parquet",
        source_index=0,
        split="train",
        mix_occurrence=0,
    )
    assert built_math["prompt"][0] == built_search["prompt"][0]
    assert built_math["extra_info"]["expected_skill"] == CODE_SKILL.name
    assert built_search["extra_info"]["expected_skill"] == SEARCH_SKILL.name
    joined_prompt = "\n".join(message["content"] for message in built_math["prompt"])
    assert CODE_SKILL.body not in joined_prompt and SEARCH_SKILL.body not in joined_prompt
    schedule = build_train_schedule(
        {"math": [math_row], "nq": [search_row, search_row], "hotpotqa": [search_row] * 3},
        balance_strategy="equal",
        seed=DEFAULT_SEED,
    )
    counts = {name: sum(source == name for source, _, _ in schedule) for name in SOURCE_ORDER}
    assert counts == {"math": 3, "nq": 3, "hotpotqa": 3}
    LOGGER.info("dataset self-test passed")


# 数据：已生成数据目录与 manifest。算法：核对文件、Parquet 行数、三来源计数和首行 contract。
def verify_output_dir(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), f"missing manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for mixed data verification") from exc

    expected_rows = {
        "train.parquet": int(manifest["train_rows"]),
        "eval_math.parquet": int(manifest["eval_rows"]["math"]),
        "eval_nq.parquet": int(manifest["eval_rows"]["nq"]),
    }
    actual_train_counts: Counter[str] = Counter()
    for filename, expected_count in expected_rows.items():
        path = output_dir / filename
        assert path.exists(), f"missing output: {path}"
        parquet_file = pq.ParquetFile(path)
        assert parquet_file.metadata.num_rows == expected_count, (
            f"row count mismatch for {path}: {parquet_file.metadata.num_rows} != {expected_count}"
        )
        assert set(parquet_file.schema.names) >= {
            "data_source",
            "role",
            "content",
            "ability",
            "label",
            "task_type",
            "expected_skill",
        }, f"unexpected schema for {path}: {parquet_file.schema.names}"
        validated_rows = 0
        for batch in parquet_file.iter_batches(
            columns=["data_source", "prompt", "label", "extra_info"],
            batch_size=8192,
        ):
            for row in batch.to_pylist():
                source_name = str(row["data_source"])
                label_payload = json.loads(row["label"])
                extra_info = row["extra_info"]
                assert isinstance(extra_info, dict)
                task_type = str(label_payload["task_type"])
                expected_skill = (
                    CODE_SKILL.name if source_name == "math" else SEARCH_SKILL.name
                )
                expected_task_type = (
                    TaskType.MATH.value if source_name == "math" else TaskType.SEARCH.value
                )
                assert source_name in SOURCE_ORDER
                assert task_type == expected_task_type
                assert extra_info["task_type"] == expected_task_type
                assert extra_info["expected_skill"] == expected_skill
                assert extra_info["source_dataset"] == source_name
                if validated_rows < 32:
                    prompt = row["prompt"]
                    assert isinstance(prompt, list) and len(prompt) == 2
                    joined_prompt = "\n".join(str(message["content"]) for message in prompt)
                    assert CODE_SKILL.body not in joined_prompt and SEARCH_SKILL.body not in joined_prompt
                    assert f"<skill_call>{CODE_SKILL.name}</skill_call>" not in joined_prompt
                    assert f"<skill_call>{SEARCH_SKILL.name}</skill_call>" not in joined_prompt
                if filename == "train.parquet":
                    actual_train_counts[source_name] += 1
                validated_rows += 1
        assert validated_rows == expected_count

    source_counts = manifest.get("train_source_counts")
    assert isinstance(source_counts, dict) and set(source_counts) == set(SOURCE_ORDER)
    assert sum(int(value) for value in source_counts.values()) == expected_rows["train.parquet"]
    assert dict(actual_train_counts) == {
        name: int(source_counts[name]) for name in SOURCE_ORDER
    }, f"actual source counts do not match manifest: {actual_train_counts}"
    if manifest.get("balance_strategy") == "equal":
        assert len({int(value) for value in source_counts.values()}) == 1

    first_row = pq.ParquetFile(output_dir / "train.parquet").read_row_group(0).slice(0, 1).to_pylist()[0]
    assert isinstance(first_row.get("prompt"), list) and len(first_row["prompt"]) == 2
    label = json.loads(first_row["label"])
    assert label["task_type"] in {TaskType.MATH.value, TaskType.SEARCH.value}
    assert first_row["extra_info"]["expected_skill"] in {CODE_SKILL.name, SEARCH_SKILL.name}
    LOGGER.info("mixed dataset verification passed: %s", output_dir)
    return manifest


# 数据：CLI 参数。算法：定义真实三源、输出、平衡和 dry self-test 配置。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--math-train",
        type=Path,
        default=Path("data/retool_skill_protocol_v0_003/math_train.jsonl"),
    )
    parser.add_argument(
        "--math-eval",
        type=Path,
        default=Path("data/retool_skill_protocol_v0_003/math_test.jsonl"),
    )
    parser.add_argument(
        "--search-train",
        type=Path,
        default=Path("data/skill_protocol/train.parquet"),
    )
    parser.add_argument(
        "--nq-eval",
        type=Path,
        default=Path("data/skill_protocol/test_nq.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mixed_data"),
    )
    parser.add_argument("--balance-strategy", choices=["equal", "natural"], default="equal")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_EVAL_LIMIT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


# 数据：真实 Math 与 Search 输入。算法：按 data_source 拆分 NQ/HotpotQA，平衡混合并写 manifest。
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.verify_only:
        verify_output_dir(args.output_dir)
        return 0
    assert args.eval_limit > 0 and args.chunk_size > 0

    LOGGER.info("reading Math train: %s", args.math_train)
    math_train = read_jsonl(args.math_train)
    LOGGER.info("reading Search train: %s", args.search_train)
    search_train = read_parquet(args.search_train)
    nq_train = [row for row in search_train if str(row.get("data_source")) == "nq"]
    hotpotqa_train = [row for row in search_train if str(row.get("data_source")) == "hotpotqa"]
    assert nq_train and hotpotqa_train, "Search train must contain NQ and HotpotQA"
    source_rows = {"math": math_train, "nq": nq_train, "hotpotqa": hotpotqa_train}
    source_paths = {
        "math": str(args.math_train),
        "nq": str(args.search_train),
        "hotpotqa": str(args.search_train),
    }
    schedule = build_train_schedule(
        source_rows,
        balance_strategy=args.balance_strategy,
        seed=args.seed,
    )
    train_counts = {
        name: sum(source_name == name for source_name, _, _ in schedule)
        for name in SOURCE_ORDER
    }
    output_dir = args.output_dir
    train_path = output_dir / "train.parquet"
    LOGGER.info("writing mixed train: %s rows=%s counts=%s", train_path, len(schedule), train_counts)
    write_parquet_rows(
        iter_train_rows(schedule, source_rows, source_paths),
        train_path,
        expected_count=len(schedule),
        chunk_size=args.chunk_size,
        force=args.force,
    )

    math_eval_source = read_jsonl(args.math_eval)
    nq_eval_source = read_parquet(args.nq_eval)
    math_eval = build_eval_rows(
        math_eval_source,
        source_name="math",
        source_path=str(args.math_eval),
        limit=args.eval_limit,
        seed=args.seed + 101,
    )
    nq_eval = build_eval_rows(
        nq_eval_source,
        source_name="nq",
        source_path=str(args.nq_eval),
        limit=args.eval_limit,
        seed=args.seed + 102,
    )
    eval_math_path = output_dir / "eval_math.parquet"
    eval_nq_path = output_dir / "eval_nq.parquet"
    write_parquet_rows(
        math_eval,
        eval_math_path,
        expected_count=len(math_eval),
        chunk_size=args.chunk_size,
        force=args.force,
    )
    write_parquet_rows(
        nq_eval,
        eval_nq_path,
        expected_count=len(nq_eval),
        chunk_size=args.chunk_size,
        force=args.force,
    )

    manifest = {
        "producer": "mixed_data.build_mixed_dataset",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": SYSTEM_PROMPT_VERSION,
        "seed": args.seed,
        "balance_strategy": args.balance_strategy,
        "source_rows": {name: len(rows) for name, rows in source_rows.items()},
        "train_rows": len(schedule),
        "train_source_counts": train_counts,
        "eval_rows": {"math": len(math_eval), "nq": len(nq_eval)},
        "paths": {
            "train": str(train_path),
            "eval_math": str(eval_math_path),
            "eval_nq": str(eval_nq_path),
        },
        "label_schema": {"task_type": "math|search", "target": "list[str]"},
        "invariants": [
            "system prompt has exactly two semantic parts",
            "both skill descriptions are visible but neither skill body is leaked",
            "expected_skill exists only in metadata and never in model prompt",
            "Math, NQ, and HotpotQA use one prompt/label/metadata schema",
            "equal balance uses every source row before deterministic resampling",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_trace = {
        name: build_unified_row(
            rows[0],
            source_name=name,
            source_path=source_paths[name],
            source_index=0,
            split="train",
            mix_occurrence=0,
        )
        for name, rows in source_rows.items()
    }
    (output_dir / "sample_trace.json").write_text(
        json.dumps(sample_trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_output_dir(output_dir)
    LOGGER.info("mixed dataset complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
