#!/usr/bin/env bash
set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
DATA_SOURCE="${DATA_SOURCE:-kaggle}"   # kaggle | gdrive
VAL_SIZE="${VAL_SIZE:-0.1}"
TEST_SIZE="${TEST_SIZE:-0.2}"
SEED="${SEED:-42}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── 0. install dependencies ───────────────────────────────────────────────────
log "=== STEP 0: install dependencies ==="
$PYTHON -m pip install -q -r "$ROOT/requirements.txt" kagglehub

# ── 1. download data ──────────────────────────────────────────────────────────
log "=== STEP 1: download data (source=$DATA_SOURCE) ==="
if [[ "$DATA_SOURCE" == "gdrive" ]]; then
    command -v gdown >/dev/null 2>&1 || die "gdown not found — install with: pip install gdown"
    bash "$ROOT/scripts/download_data.sh"
else
    [[ -n "$PYTHON" ]] || die "python3/python not found"
    $PYTHON "$ROOT/scripts/prepare_data.py"
fi

# ── 2. create splits ──────────────────────────────────────────────────────────
log "=== STEP 2: create train/val/test splits ==="
$PYTHON "$ROOT/scripts/create_splits.py" \
    --data  "$ROOT/data/tcga_redo_mlomicZ.pkl" \
    --output "$ROOT/data/splits.json" \
    --val-size "$VAL_SIZE" \
    --test-size "$TEST_SIZE" \
    --seed "$SEED"

# ── 3. train ──────────────────────────────────────────────────────────────────
log "=== STEP 3: train UniMIMIR model ==="
$PYTHON "$ROOT/train_unimimir.py"

log "=== DONE — checkpoint saved to checkpoints/best_model.pt ==="
