#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ "${PREPARE_DATA:-0}" == "1" ]]; then
  bash "$suite_root/scripts/core/prepare_data.sh" "${DATA_SEED:-20260814}"
else
  VERIFY_ONLY=1 bash "$suite_root/scripts/core/prepare_data.sh" "${DATA_SEED:-20260814}"
fi

"$python_bin" -m compileall -q "$suite_root/code"
"$python_bin" -m "$package_name.registry" validate
"$python_bin" -m "$package_name.tests.self_test"

mapfile -d '' shell_scripts < <(find "$suite_root/scripts" -type f -name '*.sh' -print0)
for script_path in "${shell_scripts[@]}"; do
  bash -n "$script_path"
done

export ALLOW_MISSING=1
dry_run_count=0
while IFS=$'\t' read -r experiment_id _; do
  [[ "$experiment_id" == "id" ]] && continue
  bash "$suite_root/scripts/core/run_experiment.sh" "$experiment_id" --dry-run </dev/null
  dry_run_count=$((dry_run_count + 1))
done < <("$python_bin" -m "$package_name.registry" list)

echo "[PASS] registry, runtime, prepared data, shell syntax, and all dry-runs are valid. experiments=$dry_run_count scripts=${#shell_scripts[@]}"
