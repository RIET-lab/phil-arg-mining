#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/opt/extra/avijit/projects/moralkg"
PYTHON_SCRIPT="${BASE_PATH}/argmining/pipeline/phase_1_old/testing/zero_shot/test_zero_shot.py"

python3 "${PYTHON_SCRIPT}" \
  -b unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  -a "${BASE_PATH}/models/meta_Llama_3.1_8B/finetune" \
  -s "${BASE_PATH}/argmining/pipeline/prompts/zero_shot_system.txt" \
  -u "${BASE_PATH}/argmining/pipeline/prompts/zero_shot_user.txt" \
  -p "${BASE_PATH}/data/docling/cleaned/RINNEF_cleaned.txt" \
