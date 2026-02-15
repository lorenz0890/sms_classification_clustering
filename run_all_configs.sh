#!/usr/bin/env bash
set -euo pipefail

configs_dir="configs"
clean_cache="false"
clean_output="false"
genai_provider="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configs-dir)
      configs_dir="$2"
      shift 2
      ;;
    --clean-cache)
      clean_cache="true"
      shift
      ;;
    --clean-output)
      clean_output="true"
      shift
      ;;
    --genai-provider)
      genai_provider="${2,,}"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--configs-dir path] [--clean-cache] [--clean-output] [--genai-provider openai|gemini]" >&2
      exit 1
      ;;
  esac
done

if [[ "${genai_provider}" != "all" && "${genai_provider}" != "openai" && "${genai_provider}" != "gemini" ]]; then
  echo "Invalid --genai-provider value: ${genai_provider}" >&2
  exit 1
fi

if [[ "${clean_cache}" == "true" ]]; then
  python utils/clean_caches.py
fi

if [[ "${clean_output}" == "true" ]]; then
  python utils/clean_output.py
fi

if [[ ! -d "${configs_dir}" ]]; then
  echo "Config directory not found: ${configs_dir}" >&2
  exit 1
fi

shopt -s nullglob

for config in "${configs_dir}"/*.json; do
  if [[ "${genai_provider}" != "all" ]]; then
    base_name="$(basename "${config}")"
    if [[ "${base_name}" == *genai_llm_config* ]]; then
      if [[ "${genai_provider}" == "openai" && "${base_name}" == *_gemini.json ]]; then
        continue
      fi
      if [[ "${genai_provider}" == "gemini" && "${base_name}" == *_openai.json ]]; then
        continue
      fi
    fi
  fi
  echo "Running ${config}"
  python main.py --config "${config}"
  echo
done

echo "Aggregating GenAI clustering metrics"
if [[ "${genai_provider}" == "all" ]]; then
  python utils/compare_genai_configs.py
else
  python utils/compare_genai_configs.py --provider "${genai_provider}"
fi
