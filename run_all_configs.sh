#!/usr/bin/env bash
set -euo pipefail

configs_dir="${1:-configs}"

if [[ ! -d "${configs_dir}" ]]; then
  echo "Config directory not found: ${configs_dir}" >&2
  exit 1
fi

shopt -s nullglob

for config in "${configs_dir}"/*.json; do
  echo "Running ${config}"
  python main.py --config "${config}"
  echo
 done
