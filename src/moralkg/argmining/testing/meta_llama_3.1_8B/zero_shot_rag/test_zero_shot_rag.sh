#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/opt/extra/avijit/projects/moralkg"
PYTHON_SCRIPT="${BASE_PATH}/src/moralkg/argmining/testing/meta_llama_3.1_8B/zero_shot_rag/test_zero_shot_rag.py"

# Defaults come from config.yaml; override here only if needed
python3 "${PYTHON_SCRIPT}" \
  -b "unsloth/Meta-Llama-3.1-8B-bnb-4bit" \
  -a "/models/meta_llama_3.1_8B/finetune" \
  -s "${BASE_PATH}/models/meta_llama_3.1_8B/prompts/rag_zero_shot_system.txt" \
  -u "${BASE_PATH}/models/meta_llama_3.1_8B/prompts/rag_zero_shot_user.txt" \
  -p "${BASE_PATH}/data/docling/cleaned/RINNEF_cleaned.txt" \
  --temperature 0.7 \
  --max-new-tokens 4096


