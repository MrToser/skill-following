#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

seed="${1:-20260814}"
if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
  exec "$python_bin" -m "$package_name.data.prepare" --root "$root_path" --seed "$seed" --verify-only
fi
exec "$python_bin" -m "$package_name.data.prepare" --root "$root_path" --seed "$seed"
