# LOCKED: false
"""从现有统一数据生成本实验套件的固定训练与评测视图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import SYSTEM_PROMPT_VERSION, TaskType, parse_training_label
from ..logger import my_logger
from ..prompts import build_prompt_messages
from ..registry import DEFAULT_ROOT, SUITE_ROOT


LOGGER = my_logger("skill_following_experiment_suite.prepare_data")
DEFAULT_SEED = 20260814
SEARCH_DATASETS = (
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
)
MATH_DATASETS = ("math", "gsm8k")
MATH_TRAIN_ROWS = 7_500
GSM8K_TRAIN_ROWS = 7_473
TABLE1_MATH_ROWS = MATH_TRAIN_ROWS + GSM8K_TRAIN_ROWS
SEARCH_R1_TRAIN_ROWS = 169_615
TABLE1_JOINT_ROWS = 400 * 72
TABLE1_JOINT_SEARCH_ROWS = TABLE1_JOINT_ROWS - TABLE1_MATH_ROWS
BOXED_START = re.compile(r"\\(?:boxed|fbox)\s*\{")


# 数据：任意 pandas/NumPy 值。算法：递归转成 JSON 安全结构供 manifest 使用。
def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


# 数据：本地文件。算法：流式计算 SHA-256，避免把大 parquet 全部载入内存。
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 数据：MATH solution。算法：从最后一个配平的 boxed/fbox 表达式中恢复答案。
def extract_last_boxed_answer(solution: Any) -> str:
    assert solution is not None
    text = str(solution)
    matches = list(BOXED_START.finditer(text))
    if not matches:
        loose_matches = list(re.finditer(r"\\boxed\s+([^$\s.,;]+)", text))
        assert loose_matches, "MATH solution has no boxed answer"
        answer = loose_matches[-1].group(1).strip()
        assert answer
        return answer
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
                    if not answer:
                        lowered = text.lower()
                        assert (
                            "no prime" in lowered
                            or "not prime for any" in lowered
                            or ("boxed{}" in lowered and "primes" in lowered and "composite" in lowered)
                        )
                        return "0"
                    return answer
    raise AssertionError("MATH solution has an unbalanced boxed answer")


# 数据：标准化数学题及来源字段。算法：构造统一 Math/Joint prompt、label 和审计元数据。
def build_math_training_row(
    *,
    question: Any,
    answer: Any,
    data_source: str,
    source_file: Path,
    source_index: int,
    task_mode: str,
    profile_name: str,
) -> dict[str, Any]:
    normalized_question = str(question).strip()
    normalized_answer = str(answer).strip()
    assert normalized_question and normalized_answer
    return {
        "data_source": data_source,
        "prompt": build_prompt_messages(normalized_question, task_mode),
        "ability": "math",
        "label": json.dumps(
            {"target": [normalized_answer], "task_type": TaskType.MATH.value},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "extra_info": {
            "task_type": TaskType.MATH.value,
            "expected_skill": "code-interpreter-protocol",
            "environment_adapter": "python_sandbox",
            "reward_source": "math_dapo",
            "answer": normalized_answer,
            "split": "train",
            "source_file": str(source_file),
            "source_dataset": data_source,
            "source_index": source_index,
            "prompt_version": SYSTEM_PROMPT_VERSION,
            "suite_data_profile": profile_name,
            "available_skill_mode": task_mode,
        },
    }


# 数据：原始 Hendrycks MATH 与 GSM8K train。算法：各读取一次并生成 14,973 道无重复数学题。
def build_exact_math_training(root: Path, *, task_mode: str, profile_name: str) -> pd.DataFrame:
    math_root = Path(os.environ.get("SF_SOURCE_DATA_ROOT", str(root / "data"))) / "01_source_data/raw/hendrycks_math"
    math_files = sorted(math_root.glob("*/train-*.parquet"))
    assert math_files
    rows: list[dict[str, Any]] = []
    math_count = 0
    for path in math_files:
        frame = pd.read_parquet(path)
        assert {"problem", "solution"}.issubset(frame.columns)
        for source_index, row in frame.iterrows():
            rows.append(
                build_math_training_row(
                    question=row["problem"],
                    answer=extract_last_boxed_answer(row["solution"]),
                    data_source="math",
                    source_file=path,
                    source_index=int(source_index),
                    task_mode=task_mode,
                    profile_name=profile_name,
                )
            )
            math_count += 1
    gsm8k_path = Path(os.environ.get("SF_SOURCE_DATA_ROOT", str(root / "data"))) / "01_source_data/raw/gsm8k/train.parquet"
    gsm8k_frame = pd.read_parquet(gsm8k_path)
    assert {"question", "label"}.issubset(gsm8k_frame.columns)
    for source_index, row in gsm8k_frame.iterrows():
        rows.append(
            build_math_training_row(
                question=row["question"],
                answer=row["label"],
                data_source="gsm8k",
                source_file=gsm8k_path,
                source_index=int(source_index),
                task_mode=task_mode,
                profile_name=profile_name,
            )
        )
    assert math_count == MATH_TRAIN_ROWS
    assert len(gsm8k_frame) == GSM8K_TRAIN_ROWS
    assert len(rows) == TABLE1_MATH_ROWS
    return pd.DataFrame(rows, columns=("data_source", "prompt", "ability", "label", "extra_info"))


# 数据：统一 prompt messages。算法：从 user 消息移除固定 suffix 后恢复原问题。
def extract_question(prompt: Any) -> str:
    messages = json_safe(prompt)
    assert isinstance(messages, list) and messages
    user_messages = [item for item in messages if isinstance(item, dict) and item.get("role") == "user"]
    assert len(user_messages) == 1
    content = str(user_messages[0].get("content") or "").strip()
    assert content.startswith("Question: ")
    question = content[len("Question: ") :].split(
        "\n\nSelect and load the useful skill before reasoning or answering.",
        1,
    )[0].strip()
    assert question
    return question


# 数据：一行统一样本。算法：从 label 获取 task_type，拒绝依赖含糊 data_source 猜测。
def row_task_type(row: pd.Series) -> str:
    return parse_training_label(row["label"]).task_type


# 数据：原始 dataframe 与可见 skill 范围。算法：保持 label 和题目不变，只重建 prompt 与审计元数据。
def rewrite_frame(frame: pd.DataFrame, *, task_mode: str, profile_name: str) -> pd.DataFrame:
    assert task_mode in {"math", "search", "joint"}
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        task_type = row_task_type(row)
        assert task_mode == "joint" or task_type == task_mode
        payload = {column: json_safe(row[column]) for column in frame.columns}
        payload["prompt"] = build_prompt_messages(extract_question(row["prompt"]), task_mode)
        metadata = payload.get("extra_info")
        assert isinstance(metadata, dict)
        metadata = dict(metadata)
        metadata.update(
            {
                "task_type": task_type,
                "prompt_version": SYSTEM_PROMPT_VERSION,
                "suite_data_profile": profile_name,
                "available_skill_mode": task_mode,
            }
        )
        payload["extra_info"] = metadata
        rows.append(payload)
    output = pd.DataFrame(rows, columns=list(frame.columns))
    assert len(output) == len(frame)
    return output


# 数据：题目行、数据集名和 seed。算法：按稳定哈希选择固定子集而不依赖原文件顺序。
def deterministic_subset(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    assert limit > 0
    keyed: list[tuple[str, int]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        metadata = json_safe(row.get("extra_info"))
        source_index = metadata.get("source_index", position) if isinstance(metadata, dict) else position
        key = hashlib.sha256(
            f"{dataset_name}|{source_index}|{seed}".encode("utf-8")
        ).hexdigest()
        keyed.append((key, position))
    selected = [position for _, position in sorted(keyed)[: min(limit, len(keyed))]]
    return frame.iloc[selected].reset_index(drop=True)


# 数据：扩增后的统一 Search train。算法：按原始题目标识去重，恢复 SearchR1 的 169,615 道训练题。
def build_unique_search_training(train_frame: pd.DataFrame) -> pd.DataFrame:
    task_types = train_frame["label"].map(lambda value: parse_training_label(value).task_type)
    search_frame = train_frame.loc[task_types == TaskType.SEARCH.value].reset_index(drop=True)
    unique_positions: list[int] = []
    seen: set[tuple[str, str, int]] = set()
    for position, metadata_value in enumerate(search_frame["extra_info"]):
        metadata = json_safe(metadata_value)
        assert isinstance(metadata, dict)
        key = (
            str(metadata.get("source_dataset") or search_frame.iloc[position]["data_source"]),
            str(metadata.get("source") or ""),
            int(metadata.get("source_index", position)),
        )
        if key not in seen:
            seen.add(key)
            unique_positions.append(position)
    unique_search = search_frame.iloc[unique_positions].reset_index(drop=True)
    assert len(unique_search) == SEARCH_R1_TRAIN_ROWS
    return unique_search


# 数据：去重后的 SearchR1 train。算法：均衡抽取 NQ 与 HotpotQA 供 400 轮 Joint 使用。
def build_joint_search_subset(unique_search: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    dataset_names = unique_search["extra_info"].map(
        lambda value: str(json_safe(value).get("source_dataset") or "")
    )
    nq_limit = TABLE1_JOINT_SEARCH_ROWS // 2
    hotpotqa_limit = TABLE1_JOINT_SEARCH_ROWS - nq_limit
    nq = deterministic_subset(
        unique_search.loc[dataset_names == "nq"].reset_index(drop=True),
        dataset_name="joint_train_nq",
        limit=nq_limit,
        seed=seed,
    )
    hotpotqa = deterministic_subset(
        unique_search.loc[dataset_names == "hotpotqa"].reset_index(drop=True),
        dataset_name="joint_train_hotpotqa",
        limit=hotpotqa_limit,
        seed=seed,
    )
    assert len(nq) == nq_limit and len(hotpotqa) == hotpotqa_limit
    selected = pd.concat([nq, hotpotqa], ignore_index=True)
    assert len(selected) == TABLE1_JOINT_SEARCH_ROWS
    return rewrite_frame(selected, task_mode="joint", profile_name="table1_joint_rerun")


# 数据：Joint train 行。算法：按来源和 seed 的稳定哈希交织 Math 与 Search。
def deterministic_order(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    keyed: list[tuple[str, int]] = []
    for position, row in frame.iterrows():
        metadata = json_safe(row["extra_info"])
        assert isinstance(metadata, dict)
        identity = "|".join(
            (
                str(row["data_source"]),
                str(metadata.get("source_file") or metadata.get("source") or ""),
                str(metadata.get("source_index", position)),
                str(seed),
            )
        )
        keyed.append((hashlib.sha256(identity.encode("utf-8")).hexdigest(), position))
    positions = [position for _, position in sorted(keyed)]
    return frame.iloc[positions].reset_index(drop=True)


# 数据：目标 parquet 与 dataframe。算法：先写临时文件再原子替换。
def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


# 数据：profile 名、数据文件与输出路径。算法：生成 slime 结构化 eval config。
def write_eval_config(
    profile_name: str,
    dataset_paths: dict[str, Path],
    output_path: Path,
    *,
    max_response_len: int,
) -> None:
    lines = [
        "# LOCKED: false",
        "eval:",
        "  defaults:",
        "    input_key: prompt",
        "    label_key: label",
        "    metadata_key: extra_info",
        "    apply_chat_template: true",
        "    n_samples_per_eval_prompt: 1",
        "    max_prompt_len: 4096",
        f"    max_response_len: {max_response_len}",
        "    custom_rm_path: skill_following.entrypoint.reward_func",
        "  datasets:",
    ]
    for dataset_name, path in dataset_paths.items():
        lines.extend(
            [
                f"    - name: {dataset_name}",
                f"      path: {path}",
                f"      metadata_overrides:",
                f"        suite_eval_profile: {profile_name}",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 数据：源数据根目录。算法：返回本地已有统一 train 和九个正式 eval 文件。
def source_catalog(root: Path) -> tuple[Path, dict[str, Path]]:
    train_path = Path(os.environ.get("SF_SOURCE_DATA_ROOT", str(root / "data"))) / "02_prepared_data/unified_search_code_v0_001/train.parquet"
    eval_root = Path(os.environ.get("SF_SOURCE_DATA_ROOT", str(root / "data"))) / "02_prepared_data/main_table_eval_v0_005/joint_ours_500step_full_dataset"
    eval_paths = {
        dataset: eval_root / f"eval_{dataset}.parquet"
        for dataset in (*MATH_DATASETS, *SEARCH_DATASETS)
    }
    assert train_path.is_file(), f"missing unified train data: {train_path}"
    missing = [path for path in eval_paths.values() if not path.is_file()]
    assert not missing, f"missing eval sources: {missing}"
    return train_path, eval_paths


# 数据：现有统一数据与目标 data 目录。算法：生成六种 train、九种实体 eval profile 和一种复用实体数据的消融 profile。
def prepare(root: Path, output_root: Path, *, seed: int) -> dict[str, Any]:
    train_source, eval_sources = source_catalog(root)
    train_frame = pd.read_parquet(train_source)
    task_types = train_frame["label"].map(lambda value: parse_training_label(value).task_type)
    table1_math = build_exact_math_training(
        root,
        task_mode="math",
        profile_name="table1_math_rerun",
    )
    table1_joint_math = rewrite_frame(
        table1_math,
        task_mode="joint",
        profile_name="table1_joint_rerun",
    )
    unique_search = build_unique_search_training(train_frame)
    table1_search = rewrite_frame(
        unique_search,
        task_mode="search",
        profile_name="table1_search_r1_rerun",
    )
    table1_joint_search = build_joint_search_subset(unique_search, seed=seed)
    table1_joint = deterministic_order(
        pd.concat([table1_joint_math, table1_joint_search], ignore_index=True),
        seed=seed,
    )
    assert len(table1_joint) == TABLE1_JOINT_ROWS
    train_views = {
        "math": rewrite_frame(
            train_frame.loc[task_types == TaskType.MATH.value].reset_index(drop=True),
            task_mode="math",
            profile_name="main_math",
        ),
        "search": rewrite_frame(
            train_frame.loc[task_types == TaskType.SEARCH.value].reset_index(drop=True),
            task_mode="search",
            profile_name="main_search",
        ),
        "joint": rewrite_frame(
            train_frame.reset_index(drop=True),
            task_mode="joint",
            profile_name="main_joint",
        ),
        "table1_math_14973": table1_math,
        "table1_search_r1_169615": table1_search,
        "table1_joint_28800": table1_joint,
    }
    manifest: dict[str, Any] = {
        "suite_version": "skill_following",
        "seed": seed,
        "source_train": {
            "path": str(train_source),
            "sha256": file_sha256(train_source),
            "rows": len(train_frame),
        },
        "train": {},
        "eval": {},
    }
    for name, frame in train_views.items():
        path = output_root / "train" / f"{name}.parquet"
        write_parquet(frame, path)
        manifest["train"][name] = {
            "path": str(path),
            "rows": len(frame),
            "sha256": file_sha256(path),
            "task_counts": frame["label"].map(
                lambda value: parse_training_label(value).task_type
            ).value_counts().sort_index().to_dict(),
        }

    eval_frames = {name: pd.read_parquet(path) for name, path in eval_sources.items()}
    profiles: dict[str, tuple[str, dict[str, pd.DataFrame]]] = {
        "main_math": ("math", {name: eval_frames[name] for name in MATH_DATASETS}),
        "main_search": ("search", {name: eval_frames[name] for name in SEARCH_DATASETS}),
        "main_joint": ("joint", {name: eval_frames[name] for name in (*MATH_DATASETS, *SEARCH_DATASETS)}),
        "monitor_math_500": (
            "math",
            {
                "math": deterministic_subset(
                    eval_frames["math"], dataset_name="monitor_math", limit=250, seed=seed
                ),
                "gsm8k": deterministic_subset(
                    eval_frames["gsm8k"], dataset_name="monitor_gsm8k", limit=250, seed=seed
                ),
            },
        ),
        "monitor_search_500": (
            "search",
            {
                "nq": deterministic_subset(
                    eval_frames["nq"], dataset_name="monitor_search_nq", limit=250, seed=seed
                ),
                "musique": deterministic_subset(
                    eval_frames["musique"], dataset_name="monitor_search_musique", limit=250, seed=seed
                ),
            },
        ),
        "monitor_joint_500": (
            "joint",
            {
                name: deterministic_subset(
                    eval_frames[name], dataset_name=f"monitor_joint_{name}", limit=125, seed=seed
                )
                for name in ("math", "gsm8k", "nq", "hotpotqa")
            },
        ),
        "rq_search": (
            "search",
            {
                "nq": deterministic_subset(eval_frames["nq"], dataset_name="nq", limit=500, seed=seed),
                "2wikimultihopqa": deterministic_subset(
                    eval_frames["2wikimultihopqa"],
                    dataset_name="2wikimultihopqa",
                    limit=500,
                    seed=seed,
                ),
            },
        ),
        "skill_screen": (
            "joint",
            {
                name: deterministic_subset(eval_frames[name], dataset_name=name, limit=100, seed=seed)
                for name in ("math", "gsm8k", "nq", "hotpotqa")
            },
        ),
        "skill_full": (
            "joint",
            {name: eval_frames[name] for name in ("math", "gsm8k", "nq", "hotpotqa")},
        ),
    }
    for profile_name, (task_mode, datasets) in profiles.items():
        profile_root = output_root / "eval" / profile_name
        output_paths: dict[str, Path] = {}
        manifest["eval"][profile_name] = {"task_mode": task_mode, "datasets": {}}
        for dataset_name, frame in datasets.items():
            rewritten = rewrite_frame(
                frame.reset_index(drop=True),
                task_mode=task_mode,
                profile_name=profile_name,
            )
            output_path = profile_root / f"eval_{dataset_name}.parquet"
            write_parquet(rewritten, output_path)
            output_paths[dataset_name] = output_path
            manifest["eval"][profile_name]["datasets"][dataset_name] = {
                "source": str(eval_sources[dataset_name]),
                "source_sha256": file_sha256(eval_sources[dataset_name]),
                "path": str(output_path),
                "rows": len(rewritten),
                "sha256": file_sha256(output_path),
            }
        max_response_len = 2048 if task_mode == "search" else 8192
        write_eval_config(
            profile_name,
            output_paths,
            profile_root / "eval_config.yaml",
            max_response_len=max_response_len,
        )
        (profile_root / "runtime_config.yaml").write_text(
            f"# LOCKED: false\nmax_turns: {8 if task_mode == 'search' else 20}\n",
            encoding="utf-8",
        )

    # 数据：main_search 中的完整 2Wiki 与 HotpotQA 文件。算法：复用同一实体数据，只生成本轮推理消融所需的窄评测视图。
    ablation_profile = "ablation_search_2wiki_hotpotqa_full"
    ablation_datasets = ("2wikimultihopqa", "hotpotqa")
    ablation_root = output_root / "eval" / ablation_profile
    ablation_paths = {
        dataset_name: output_root / "eval" / "main_search" / f"eval_{dataset_name}.parquet"
        for dataset_name in ablation_datasets
    }
    write_eval_config(
        ablation_profile,
        ablation_paths,
        ablation_root / "eval_config.yaml",
        max_response_len=2048,
    )
    (ablation_root / "runtime_config.yaml").write_text(
        "# LOCKED: false\nmax_turns: 8\n",
        encoding="utf-8",
    )
    manifest["eval"][ablation_profile] = {
        "task_mode": "search",
        "datasets": {
            dataset_name: dict(manifest["eval"]["main_search"]["datasets"][dataset_name])
            for dataset_name in ablation_datasets
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "prepared train rows math=%s search=%s joint=%s table1_math=%s table1_search=%s table1_joint=%s and %s eval profiles",
        len(train_views["math"]),
        len(train_views["search"]),
        len(train_views["joint"]),
        len(train_views["table1_math_14973"]),
        len(train_views["table1_search_r1_169615"]),
        len(train_views["table1_joint_28800"]),
        len(profiles) + 1,
    )
    return manifest


# 数据：已生成 manifest。算法：重算所有输出的行数和 SHA-256。
def verify(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    entries = list(manifest["train"].values())
    for profile in manifest["eval"].values():
        entries.extend(profile["datasets"].values())
    for entry in entries:
        path = Path(entry["path"])
        assert path.is_file(), f"missing prepared data: {path}"
        assert len(pd.read_parquet(path)) == int(entry["rows"])
        assert file_sha256(path) == entry["sha256"]
        checked += 1
    table1_math = pd.read_parquet(manifest["train"]["table1_math_14973"]["path"])
    table1_search = pd.read_parquet(manifest["train"]["table1_search_r1_169615"]["path"])
    table1_joint = pd.read_parquet(manifest["train"]["table1_joint_28800"]["path"])
    assert table1_math["data_source"].value_counts().to_dict() == {
        "math": MATH_TRAIN_ROWS,
        "gsm8k": GSM8K_TRAIN_ROWS,
    }
    assert table1_joint["data_source"].value_counts().to_dict() == {
        "math": MATH_TRAIN_ROWS,
        "gsm8k": GSM8K_TRAIN_ROWS,
        "hotpotqa": TABLE1_JOINT_SEARCH_ROWS - TABLE1_JOINT_SEARCH_ROWS // 2,
        "nq": TABLE1_JOINT_SEARCH_ROWS // 2,
    }
    assert table1_search["data_source"].value_counts().to_dict() == {
        "hotpotqa": 90_447,
        "nq": 79_168,
    }
    joint_task_counts = table1_joint["label"].map(
        lambda value: parse_training_label(value).task_type
    ).value_counts().to_dict()
    assert joint_task_counts == {
        TaskType.MATH.value: TABLE1_MATH_ROWS,
        TaskType.SEARCH.value: TABLE1_JOINT_SEARCH_ROWS,
    }
    assert {
        name: entry["rows"]
        for name, entry in manifest["eval"]["monitor_math_500"]["datasets"].items()
    } == {"math": 250, "gsm8k": 250}
    assert {
        name: entry["rows"]
        for name, entry in manifest["eval"]["monitor_joint_500"]["datasets"].items()
    } == {"math": 125, "gsm8k": 125, "nq": 125, "hotpotqa": 125}
    assert {
        name: entry["rows"]
        for name, entry in manifest["eval"]["monitor_search_500"]["datasets"].items()
    } == {"nq": 250, "musique": 250}
    assert {
        name: entry["rows"]
        for name, entry in manifest["eval"]["main_math"]["datasets"].items()
    } == {"math": 5_000, "gsm8k": 1_319}
    LOGGER.info("verified %s prepared parquet files", checked)
    return manifest


# 数据：命令行路径和 seed。算法：prepare 后强制 verify，或只验证现有产物。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=SUITE_ROOT / "data" / "prepared")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        verify(args.output_root)
        return 0
    prepare(args.root.resolve(), args.output_root.resolve(), seed=args.seed)
    verify(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
