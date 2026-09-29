#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

experiment_id="${1:?usage: run_experiment.sh EXPERIMENT_ID [--phase train|eval] [--seed N] [--dry-run]}"
shift
phase=""
seed=20260814
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      phase="${2:?missing phase}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing seed}"
      shift 2
      ;;
    --dry-run)
      export DRY_RUN=1
      shift
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

kind="$($python_bin -m "$package_name.registry" get "$experiment_id" kind)"
if [[ -z "$phase" ]]; then
  case "$kind" in
    train) phase=train ;;
    eval) phase=eval ;;
    *) echo "[ERROR] unsupported kind=$kind" >&2; exit 1 ;;
  esac
fi

case "$phase" in
  train)
    exec bash "$suite_root/scripts/core/run_train.sh" "$experiment_id" "$seed"
    ;;
  eval)
    exec bash "$suite_root/scripts/core/run_eval.sh" "$experiment_id" "$seed"
    ;;
  *)
    echo "[ERROR] phase must be train or eval" >&2
    exit 1
    ;;
esac
