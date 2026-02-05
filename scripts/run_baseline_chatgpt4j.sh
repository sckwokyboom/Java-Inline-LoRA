#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# --- uv bootstrap (user-local, no sudo) ---
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to user environment..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # ensure new shell can find uv (typical install path)
  export PATH="$HOME/.local/bin:$PATH"
fi

VENV_PATH="${VENV_PATH:-.venv}"

# Create/manage venv with uv (recommended) OR keep your python -m venv.
# Using uv is more consistent:
uv venv --path "${VENV_PATH}" >/dev/null

# Activate venv (so 'python' points inside it for your scripts)
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

# Sync dependencies from pyproject.toml (and uv.lock if present)
# --frozen: fail if lock exists but doesn't match (good for reproducibility)
# If you don't use uv.lock yet, you can drop --frozen.
if [ -f "uv.lock" ]; then
  uv sync --frozen
else
  uv sync
fi

# Install your package editable into the venv (so `import ...` works)
uv pip install -e .

DATA_REPOS_DIR="${REPO_ROOT}/data/repos"
TARGET_REPO="${DATA_REPOS_DIR}/chatgpt4j"
mkdir -p "${DATA_REPOS_DIR}"
if [ ! -d "${TARGET_REPO}/.git" ]; then
  git clone https://github.com/itlemon/chatgpt4j "${TARGET_REPO}"
else
  git -C "${TARGET_REPO}" pull --ff-only
fi

TRAIN_PATH="${REPO_ROOT}/data/chatgpt4j/train.jsonl"
VAL_PATH="${REPO_ROOT}/data/chatgpt4j/val.jsonl"

python scripts/make_dataset.py \
  --repo "${TARGET_REPO}" \
  --include_header \
  --split_by_file \
  --max_prefix_lines 80 \
  --max_suffix_lines 80 \
  --max_samples_per_file 80 \
  --val_ratio 0.02 \
  --out_train "${TRAIN_PATH}" \
  --out_val "${VAL_PATH}"

python - <<'PY'
from itertools import islice
from pathlib import Path

def cap(path: str, limit: int) -> None:
    p = Path(path)
    if not p.exists():
        return
    lines = list(islice(p.open("r", encoding="utf-8"), limit))
    if lines:
        p.write_text("".join(lines), encoding="utf-8")

cap("data/chatgpt4j/train.jsonl", 1000)
cap("data/chatgpt4j/val.jsonl", 1000)
print("Trimmed train/val to at most 1000 samples each.")
PY

python scripts/train_lora.py \
  --model /mnt/data/models/Qwen2.5-Coder-0.5B \
  --train "${TRAIN_PATH}" \
  --val "${VAL_PATH}" \
  --out adapters/chatgpt4j-qwen25coder0_5b-lora \
  --max_length 2048 \
  --epochs 0.2 \
  --batch_size 2 \
  --grad_accum 1 \
  --lr 2e-4 \
  --logging_steps 20 \
  --eval_steps 200 \
  --save_steps 200

echo "Baseline training finished. Adapter saved to ${REPO_ROOT}/adapters/chatgpt4j-qwen25coder0_5b-lora"