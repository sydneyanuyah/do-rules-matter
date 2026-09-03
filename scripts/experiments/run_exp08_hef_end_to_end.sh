#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-${PAPER1_PROJECT_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/exp08/hef_end_to_end}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export USE_TF=0

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_exp08_hef_end_to_end.py" \
  --project-root "${PROJECT_ROOT}" \
  --device cuda \
  --batch-size "${EXP08_BATCH_SIZE:-512}" \
  --warmups "${EXP08_WARMUPS:-1}" \
  --repeats "${EXP08_REPEATS:-3}" \
  --online-pairs "${EXP08_ONLINE_PAIRS:-100}" \
  --output "${OUTPUT_ROOT}"
