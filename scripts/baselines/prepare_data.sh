#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_env harnessr1
prepare_baseline_data
"$python_bin" -m skill_following_baselines.self_test
