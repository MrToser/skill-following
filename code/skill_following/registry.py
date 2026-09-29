# LOCKED: false
"""单一实验注册表及其命令行查询接口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import OBSERVATION_MODES, SKILL_VARIANTS, TASK_MODES
from .logger import my_logger


LOGGER = my_logger("skill_following_experiment_suite.registry")
PACKAGE_DIR = Path(__file__).resolve().parent
SUITE_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_ROOT = Path(os.environ.get("ROOT_PATH", Path.cwd()))
REGISTRY_PATH = SUITE_ROOT / "code" / "config" / "experiments.json"
VALID_KINDS = {"train", "eval", "alias", "legacy", "analysis"}
VALID_PRIORITIES = {"P0", "P1", "P2"}


# 数据：实验默认值、可复用训练 preset 与单条实验。算法：按默认值、preset、实验覆盖的顺序合并。
def merge_experiment(registry: dict[str, Any], raw_experiment: dict[str, Any]) -> dict[str, Any]:
    preset_name = raw_experiment.get("training_preset")
    presets = registry.get("training_presets") or {}
    if preset_name is not None and preset_name not in presets:
        raise KeyError(f"unknown training_preset={preset_name} for {raw_experiment.get('id')}")
    preset = presets.get(preset_name) or {}
    return {**registry.get("experiment_defaults", {}), **preset, **raw_experiment}


# 数据：带 ROOT/SUITE/SEED 占位符的注册表值。算法：递归展开且不执行 shell。
def expand_value(value: Any, *, root: Path, seed: int) -> Any:
    if isinstance(value, str):
        return (
            value.replace("{ROOT}", str(root))
            .replace("{SUITE}", str(SUITE_ROOT))
            .replace("{SEED}", str(seed))
        )
    if isinstance(value, list):
        return [expand_value(item, root=root, seed=seed) for item in value]
    if isinstance(value, dict):
        return {
            key: expand_value(item, root=root, seed=seed)
            for key, item in value.items()
        }
    return value


# 数据：JSON 注册表路径。算法：读取后立即做结构和实验不变量检查。
def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(payload)
    return payload


# 数据：完整注册表。算法：验证 ID、模型、profile、干预值和 checkpoint 依赖。
def validate_registry(registry: dict[str, Any]) -> None:
    assert registry.get("suite_version") == "skill_following"
    status_snapshot = registry.get("status_snapshot")
    assert isinstance(status_snapshot, dict)
    assert str(status_snapshot.get("updated_at") or "").strip()
    assert str(status_snapshot.get("source") or "").strip()
    assert str(status_snapshot.get("note") or "").strip()
    models = registry.get("models")
    profiles = registry.get("data_profiles")
    presets = registry.get("training_presets")
    experiments = registry.get("experiments")
    assert isinstance(models, dict) and models
    assert isinstance(profiles, dict) and profiles
    assert isinstance(presets, dict) and presets
    assert all(name.strip() and isinstance(preset, dict) for name, preset in presets.items())
    assert isinstance(experiments, list) and experiments
    experiment_ids = [str(item.get("id") or "") for item in experiments]
    assert all(experiment_ids) and len(experiment_ids) == len(set(experiment_ids))
    known_ids = set(experiment_ids)
    for model_name, model in models.items():
        assert model_name.strip()
        for field in ("hf_repo", "model_config", "hf_checkpoint", "ref_load"):
            assert str(model.get(field) or "").strip()
        assert int(model.get("train_gpus") or 0) > 0
        assert int(model.get("train_tp") or 0) > 0
        assert int(model.get("eval_gpus") or 0) > 0
        assert int(model.get("eval_tp") or 0) > 0
    for raw_experiment in experiments:
        experiment = merge_experiment(registry, raw_experiment)
        assert experiment["kind"] in VALID_KINDS
        assert experiment["priority"] in VALID_PRIORITIES
        assert str(experiment["status"]).strip()
        assert experiment["task_mode"] in TASK_MODES
        assert experiment["skill_variant"] in SKILL_VARIANTS
        assert experiment["observation_mode"] in OBSERVATION_MODES
        assert isinstance(experiment["state_aware_hints"], bool)
        assert isinstance(experiment["transition_credit"], bool)
        assert isinstance(experiment["skip_eval_before_train"], bool)
        assert 0.0 < float(experiment["recovery_factor"]) <= 1.0
        assert 0.0 <= float(experiment["first_research_reward"]) <= 0.2
        assert experiment["model"] in models
        assert all(isinstance(seed, int) and seed >= 0 for seed in experiment["seeds"])
        for field in (
            "rollout_batch_size",
            "n_samples_per_prompt",
            "num_steps_per_rollout",
            "global_batch_size",
            "eval_interval",
            "save_interval",
            "rollout_max_response_len",
            "eval_max_response_len",
            "sglang_server_concurrency",
        ):
            assert int(experiment[field]) > 0, f"{experiment['id']} has invalid {field}"
        assert (
            int(experiment["rollout_batch_size"])
            * int(experiment["n_samples_per_prompt"])
            == int(experiment["global_batch_size"])
            * int(experiment["num_steps_per_rollout"])
        ), f"{experiment['id']} has inconsistent rollout and optimizer batch sizes"
        if experiment.get("train_tp") is not None:
            assert int(experiment["train_tp"]) > 0
        if experiment.get("train_gpus") is not None:
            assert int(experiment["train_gpus"]) > 0
        if experiment.get("eval_tp") is not None:
            assert int(experiment["eval_tp"]) > 0
        if experiment.get("eval_gpus") is not None:
            assert int(experiment["eval_gpus"]) > 0
        selected_eval_gpus = int(
            experiment.get("eval_gpus") or models[experiment["model"]]["eval_gpus"]
        )
        selected_eval_tp = int(
            experiment.get("eval_tp") or models[experiment["model"]]["eval_tp"]
        )
        assert selected_eval_gpus % selected_eval_tp == 0, (
            f"{experiment['id']} has eval_gpus not divisible by eval_tp"
        )
        assert 0.0 <= float(experiment["lr_warmup_fraction"]) <= 1.0
        assert float(experiment["weight_decay"]) >= 0.0
        assert float(experiment["entropy_coef"]) >= 0.0
        assert 0.0 < float(experiment["eps_clip"]) <= float(experiment["eps_clip_high"])
        assert float(experiment["clip_grad"]) > 0.0
        assert float(experiment["kl_loss_coef"]) >= 0.0
        assert float(experiment["tis_clip"]) > 0.0
        if experiment.get("train_profile") is not None:
            assert experiment["train_profile"] in profiles
        if experiment.get("eval_profile") is not None:
            assert experiment["eval_profile"] in profiles
        if experiment.get("train_eval_profile") is not None:
            assert experiment["train_eval_profile"] in profiles
        checkpoint_from = experiment.get("checkpoint_from")
        if checkpoint_from:
            assert checkpoint_from in known_ids, (
                f"unknown checkpoint_from={checkpoint_from} for {experiment['id']}"
            )
        if experiment.get("save_path") is not None:
            assert str(experiment["save_path"]).strip()
        if experiment["kind"] == "analysis":
            assert str(experiment.get("analysis_action") or "").strip()
        if experiment["kind"] == "legacy":
            assert str(experiment.get("legacy_launcher") or "").strip()


# 数据：实验 ID。算法：合并全局默认值并返回一份独立配置。
def resolve_experiment(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    matches = [item for item in registry["experiments"] if item["id"] == experiment_id]
    if not matches:
        raise KeyError(f"unknown experiment_id: {experiment_id}")
    assert len(matches) == 1
    return merge_experiment(registry, matches[0])


# 数据：实验及 seed。算法：递归解析评测加载所需的默认 checkpoint、依赖 checkpoint 或冻结路径。
def resolve_checkpoint(
    registry: dict[str, Any],
    experiment: dict[str, Any],
    *,
    root: Path,
    seed: int,
) -> str:
    explicit = experiment.get("checkpoint") or experiment.get("save_path")
    if explicit:
        return str(expand_value(explicit, root=root, seed=seed))
    dependency = experiment.get("checkpoint_from")
    if dependency:
        source = resolve_experiment(registry, dependency)
        return resolve_checkpoint(registry, source, root=root, seed=seed)
    return str(
        SUITE_ROOT
        / "data"
        / "artifacts"
        / "checkpoints"
        / experiment["id"]
        / f"seed_{seed}"
    )


# 数据：实验 ID、运行阶段和 seed。算法：展开为 shell 与 Ray 共享的平坦环境契约。
def build_environment(
    registry: dict[str, Any],
    experiment_id: str,
    *,
    phase: str,
    seed: int,
    root: Path,
) -> dict[str, str]:
    assert phase in {"train", "eval", "analysis"}
    experiment = resolve_experiment(registry, experiment_id)
    model = expand_value(registry["models"][experiment["model"]], root=root, seed=seed)
    profiles = registry["data_profiles"]
    train_profile = profiles.get(experiment.get("train_profile") or "") or {}
    selected_eval_profile = experiment.get("eval_profile")
    if phase == "train" and experiment.get("train_eval_profile") is not None:
        selected_eval_profile = experiment["train_eval_profile"]
    eval_profile = profiles.get(selected_eval_profile or "") or {}
    train_profile = expand_value(train_profile, root=root, seed=seed)
    eval_profile = expand_value(eval_profile, root=root, seed=seed)
    checkpoint = resolve_checkpoint(registry, experiment, root=root, seed=seed)
    artifact_root = SUITE_ROOT / "data" / "artifacts" / experiment_id / f"seed_{seed}"
    selected_gpus = model["train_gpus"] if phase == "train" else model["eval_gpus"]
    selected_tp = model["train_tp"] if phase == "train" else model["eval_tp"]
    if phase == "train" and experiment.get("train_gpus") is not None:
        selected_gpus = int(experiment["train_gpus"])
    if phase == "train" and experiment.get("train_tp") is not None:
        selected_tp = int(experiment["train_tp"])
    if phase == "eval" and experiment.get("eval_gpus") is not None:
        selected_gpus = int(experiment["eval_gpus"])
    if phase == "eval" and experiment.get("eval_tp") is not None:
        selected_tp = int(experiment["eval_tp"])
    environment = {
        "SF_EXP_ID": experiment_id,
        "SF_EXP_KIND": str(experiment["kind"]),
        "SF_EXP_PRIORITY": str(experiment["priority"]),
        "SF_EXP_RQ": str(experiment["rq"]),
        "SF_EXP_RUN_PHASE": phase,
        "SF_EXP_TASK_MODE": str(experiment["task_mode"]),
        "SF_EXP_SKILL_VARIANT": str(experiment["skill_variant"]),
        "SF_EXP_STATE_AWARE_HINTS": "1" if experiment["state_aware_hints"] else "0",
        "SF_EXP_TRANSITION_CREDIT": "1" if experiment["transition_credit"] else "0",
        "SF_EXP_RECOVERY_FACTOR": str(experiment["recovery_factor"]),
        "SF_EXP_FIRST_RESEARCH_REWARD": str(experiment["first_research_reward"]),
        "SF_EXP_OBSERVATION_MODE": str(experiment["observation_mode"]),
        "SF_SKIP_EVAL_BEFORE_TRAIN": "1" if experiment["skip_eval_before_train"] else "0",
        "SF_EXP_SEED": str(seed),
        "SF_MODEL_KEY": str(experiment["model"]),
        "MODEL_CONFIG": str(model["model_config"]),
        "HF_CHECKPOINT": str(model["hf_checkpoint"]),
        "REF_LOAD": str(model["ref_load"]),
        "NUM_GPUS": str(selected_gpus),
        "TP": str(selected_tp),
        "TRAIN_DATA": str(train_profile.get("train_path") or eval_profile.get("train_path") or ""),
        "EVAL_CONFIG_PATH": str(eval_profile.get("eval_config") or ""),
        "EVAL_PROFILE": str(selected_eval_profile or ""),
        "NUM_ROLLOUT": str(experiment.get("steps") or 0),
        "ROLLOUT_BATCH_SIZE": str(experiment["rollout_batch_size"]),
        "N_SAMPLES_PER_PROMPT": str(experiment["n_samples_per_prompt"]),
        "NUM_STEPS_PER_ROLLOUT": str(experiment["num_steps_per_rollout"]),
        "GLOBAL_BATCH_SIZE": str(experiment["global_batch_size"]),
        "EVAL_INTERVAL": str(experiment["eval_interval"]),
        "SAVE_INTERVAL": str(experiment["save_interval"]),
        "ROLLOUT_MAX_RESPONSE_LEN": str(experiment["rollout_max_response_len"]),
        "EVAL_MAX_RESPONSE_LEN": str(experiment["eval_max_response_len"]),
        "SGLANG_SERVER_CONCURRENCY": str(experiment["sglang_server_concurrency"]),
        "LR_WARMUP_FRACTION": str(experiment["lr_warmup_fraction"]),
        "WEIGHT_DECAY": str(experiment["weight_decay"]),
        "ENTROPY_COEF": str(experiment["entropy_coef"]),
        "EPS_CLIP": str(experiment["eps_clip"]),
        "EPS_CLIP_HIGH": str(experiment["eps_clip_high"]),
        "CLIP_GRAD": str(experiment["clip_grad"]),
        "KL_LOSS_COEF": str(experiment["kl_loss_coef"]),
        "TIS_CLIP": str(experiment["tis_clip"]),
        "SAVE_PATH": str(
            expand_value(experiment["save_path"], root=root, seed=seed)
            if experiment.get("save_path")
            else SUITE_ROOT
            / "data"
            / "artifacts"
            / "checkpoints"
            / experiment_id
            / f"seed_{seed}"
        )
        if experiment["kind"] == "train"
        else str(artifact_root / "unused_save"),
        "CHECKPOINT": checkpoint,
        "ARTIFACT_ROOT": str(artifact_root),
        "TRACE_DIR": str(artifact_root / "traces"),
        "SF_TRACE_OUTPUT_PATH": str(artifact_root / "traces" / "train_token_trace.jsonl"),
        "SF_AUDIT_TRACE_PATH": str(artifact_root / "traces" / "train_audit_trace.jsonl"),
        "SF_EVAL_TRACE_PATH": str(artifact_root / "eval" / "samples.jsonl"),
        "SF_EVAL_SUMMARY_PATH": str(artifact_root / "eval" / "summary.json"),
        "WANDB_DIR": str(SUITE_ROOT / "data" / "artifacts" / "wandb_offline"),
        "WANDB_GROUP": f"{experiment_id}-seed-{seed}",
        "RJOB_PRIORITY": str(experiment.get("rjob_priority") or 9),
        "LEGACY_LAUNCHER": str(
            expand_value(experiment.get("legacy_launcher") or "", root=root, seed=seed)
        ),
        "ANALYSIS_ACTION": str(experiment.get("analysis_action") or ""),
    }
    for key, value in environment.items():
        if "\n" in value or "\t" in value:
            raise ValueError(f"environment value {key} contains a forbidden delimiter")
    return environment


# 数据：运行环境和阶段。算法：列出实际执行前必须存在的输入路径。
def required_paths(environment: dict[str, str], *, phase: str) -> list[Path]:
    required = [Path(environment["MODEL_CONFIG"]), Path(environment["HF_CHECKPOINT"])]
    if environment["SF_EXP_KIND"] not in {"analysis", "legacy"}:
        required.append(Path(environment["REF_LOAD"]))
    if phase == "train" and environment["SF_EXP_KIND"] != "legacy":
        required.append(Path(environment["TRAIN_DATA"]))
    if phase == "eval":
        required.extend(
            [
                Path(environment["TRAIN_DATA"]),
                Path(environment["EVAL_CONFIG_PATH"]),
                Path(environment["CHECKPOINT"]),
            ]
        )
    if environment["SF_EXP_KIND"] == "legacy":
        required.append(Path(environment["LEGACY_LAUNCHER"]))
    return [path for path in required if str(path)]


# 数据：训练 checkpoint 路径。算法：要求存在最新迭代标记和至少一个完整的分布式迭代目录，拒绝空占位目录。
def checkpoint_ready(checkpoint_path: Path) -> bool:
    if checkpoint_path.is_file():
        return True
    if not checkpoint_path.is_dir():
        return False
    latest_iteration = checkpoint_path / "latest_checkpointed_iteration.txt"
    if not latest_iteration.is_file() or not latest_iteration.read_text(encoding="utf-8").strip():
        return False
    for iteration_path in checkpoint_path.glob("iter_*"):
        if iteration_path.is_dir() and (iteration_path / ".metadata").is_file() and (
            iteration_path / "common.pt"
        ).is_file():
            return True
    return False


# 数据：Hugging Face 运行时目录。算法：在占用 GPU 前验证 AutoConfig 与 tokenizer 所需元数据完整。
def hf_runtime_metadata_ready(checkpoint_path: Path) -> bool:
    config_path = checkpoint_path / "config.json"
    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not str(config.get("model_type") or "").strip():
        return False
    tokenizer_assets = (
        checkpoint_path / "tokenizer.json",
        checkpoint_path / "tokenizer.model",
        checkpoint_path / "vocab.json",
    )
    return any(path.is_file() and path.stat().st_size > 0 for path in tokenizer_assets)


# 数据：环境字典。算法：输出无歧义 TSV，供 Bash 逐项 export。
def print_environment(environment: dict[str, str]) -> None:
    for key in sorted(environment):
        print(f"{key}\t{environment[key]}")


# 数据：环境字典与当前进程路径。算法：生成 Ray runtime-env JSON，不手拼转义。
def print_runtime_env(environment: dict[str, str]) -> None:
    forwarded = {
        key: value
        for key, value in environment.items()
        if key.startswith("SF_EXP_")
        or key.startswith("SF_TRACE_")
        or key.startswith("SF_AUDIT_")
        or key.startswith("SF_EVAL_")
    }
    for key in (
        "PYTHONPATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "UNIFIED_SEARCH_URL",
        "UNIFIED_SEARCH_TOPK",
        "UNIFIED_SEARCH_CONCURRENCY",
        "WANDB_MODE",
        "WANDB_DIR",
    ):
        value = os.environ.get(key)
        if value is not None:
            forwarded[key] = value
    print(json.dumps({"env_vars": forwarded}, separators=(",", ":"), sort_keys=True))


# 数据：注册表命令行参数。算法：统一 list/show/env/get/validate/preflight/runtime-env。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    show = subparsers.add_parser("show")
    show.add_argument("experiment_id")
    get = subparsers.add_parser("get")
    get.add_argument("experiment_id")
    get.add_argument("field")
    for name in ("env", "runtime-env", "preflight"):
        command = subparsers.add_parser(name)
        command.add_argument("experiment_id")
        command.add_argument("--phase", choices=("train", "eval", "analysis"), required=True)
        command.add_argument("--seed", type=int, default=20260814)
    subparsers.add_parser("validate")
    return parser


# 数据：解析后的 CLI 参数。算法：查询注册表或验证运行先决条件。
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry(args.registry)
    root = Path(os.environ.get("ROOT_PATH", str(DEFAULT_ROOT))).resolve()
    if args.command == "list":
        print("id\tpriority\tkind\trq\tmodel\ttask\tstatus")
        for raw in registry["experiments"]:
            experiment = merge_experiment(registry, raw)
            print(
                "\t".join(
                    str(experiment[field])
                    for field in ("id", "priority", "kind", "rq", "model", "task_mode", "status")
                )
            )
        return 0
    if args.command == "show":
        print(json.dumps(resolve_experiment(registry, args.experiment_id), ensure_ascii=False, indent=2))
        return 0
    if args.command == "get":
        experiment = resolve_experiment(registry, args.experiment_id)
        if args.field not in experiment:
            raise KeyError(f"field {args.field!r} does not exist")
        value = experiment[args.field]
        print(json.dumps(value) if isinstance(value, (dict, list, bool)) else value)
        return 0
    if args.command == "validate":
        LOGGER.info(
            "registry valid: experiments=%s models=%s profiles=%s",
            len(registry["experiments"]),
            len(registry["models"]),
            len(registry["data_profiles"]),
        )
        return 0
    environment = build_environment(
        registry,
        args.experiment_id,
        phase=args.phase,
        seed=args.seed,
        root=root,
    )
    if args.command == "env":
        print_environment(environment)
        return 0
    if args.command == "runtime-env":
        print_runtime_env(environment)
        return 0
    missing = [path for path in required_paths(environment, phase=args.phase) if not path.exists()]
    if missing:
        for path in missing:
            print(f"MISSING\t{path}")
        return 2
    if environment["SF_EXP_KIND"] not in {"analysis", "legacy"}:
        hf_checkpoint = Path(environment["HF_CHECKPOINT"])
        if not hf_runtime_metadata_ready(hf_checkpoint):
            print(f"INCOMPLETE_HF_METADATA\t{hf_checkpoint}")
            return 2
    if args.phase == "eval":
        checkpoint_path = Path(environment["CHECKPOINT"])
        if not checkpoint_ready(checkpoint_path):
            print(f"INCOMPLETE_CHECKPOINT\t{checkpoint_path}")
            return 2
    print(f"READY\t{args.experiment_id}\t{args.phase}\tseed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
