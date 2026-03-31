#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full pipeline: download → train → evaluate
#
# Usage:
#   bash run_all.sh --api-key <roboflow_key>
#   bash run_all.sh --api-key <roboflow_key> --run-name v1
#
# Steps:
#   1. Download segmentation dataset (stelar/rdd-mining-road-seg v1)
#   2. Download detection dataset    (stelar/rdd-mining-road-det  v2)
#   3. Train segmentation model      (yolov8n-seg)
#   4. Evaluate segmentation model   (day / wet / night)
#   5. Train detection model         (yolov8m)
#   6. Evaluate detection model      (day / wet / night)
#
# NOTE: Per-condition test splits (day/wet/night) must be populated manually
#       before running. The download step creates the empty dirs and reminds you.
# =============================================================================

set -euo pipefail

# Load .env if present
if [[ -f "$(dirname "$0")/.env" ]]; then
    set -a; source "$(dirname "$0")/.env"; set +a
fi

# --------------------------------------------------------------------------- #
# Parse args
# --------------------------------------------------------------------------- #
API_KEY="${ROBOFLOW_API_KEY:-}"
RUN_NAME="v1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key)  API_KEY="$2";  shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$API_KEY" ]]; then
    echo "Error: --api-key is required"
    echo "Usage: bash run_all.sh --api-key <roboflow_key> [--run-name v1]"
    exit 1
fi

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log_step() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

extract_weights() {
    # Extract path from "BEST_WEIGHTS=<path>" line printed by train.py
    grep "^BEST_WEIGHTS=" "$1" | tail -1 | cut -d'=' -f2-
}

# --------------------------------------------------------------------------- #
# Step 1 & 2 — Download datasets
# --------------------------------------------------------------------------- #
log_step "1/6  Downloading segmentation dataset"
python scripts/download_datasets.py --api-key "$API_KEY" --dataset seg

log_step "2/6  Downloading detection dataset"
python scripts/download_datasets.py --api-key "$API_KEY" --dataset det

# --------------------------------------------------------------------------- #
# Step 3 — Train segmentation
# --------------------------------------------------------------------------- #
log_step "3/6  Training segmentation model (yolov8n-seg)"
SEG_LOG="logs/train_seg_${RUN_NAME}.log"
mkdir -p logs
python train.py \
    --config config/yolo_segmentation.yaml \
    --run-name "seg-${RUN_NAME}" \
    2>&1 | tee "$SEG_LOG"

SEG_WEIGHTS=$(extract_weights "$SEG_LOG")
if [[ -z "$SEG_WEIGHTS" || ! -f "$SEG_WEIGHTS" ]]; then
    echo "Error: could not locate segmentation best.pt. Check $SEG_LOG"
    exit 1
fi
echo "Segmentation weights: $SEG_WEIGHTS"

# --------------------------------------------------------------------------- #
# Step 4 — Evaluate segmentation
# --------------------------------------------------------------------------- #
log_step "4/6  Evaluating segmentation model (day / wet / night)"
python evaluate.py \
    --config config/yolo_segmentation.yaml \
    --model "$SEG_WEIGHTS" \
    --run-name "eval-seg-${RUN_NAME}" \
    2>&1 | tee "logs/eval_seg_${RUN_NAME}.log"

# Standalone visualize — larger client-facing video set (60 frames ≈ 20s @ 3fps)
python visualize.py \
    --config config/yolo_segmentation.yaml \
    --model "$SEG_WEIGHTS" \
    --n-samples 60 \
    --fps 3 \
    --run-name "vis-seg-${RUN_NAME}" \
    2>&1 | tee "logs/vis_seg_${RUN_NAME}.log"

# --------------------------------------------------------------------------- #
# Step 5 — Train detection
# --------------------------------------------------------------------------- #
log_step "5/6  Training detection model (yolov8m)"
DET_LOG="logs/train_det_${RUN_NAME}.log"
python train.py \
    --config config/yolo_detection.yaml \
    --run-name "det-${RUN_NAME}" \
    2>&1 | tee "$DET_LOG"

DET_WEIGHTS=$(extract_weights "$DET_LOG")
if [[ -z "$DET_WEIGHTS" || ! -f "$DET_WEIGHTS" ]]; then
    echo "Error: could not locate detection best.pt. Check $DET_LOG"
    exit 1
fi
echo "Detection weights: $DET_WEIGHTS"

# --------------------------------------------------------------------------- #
# Step 6 — Evaluate detection
# --------------------------------------------------------------------------- #
log_step "6/6  Evaluating detection model (day / wet / night)"
python evaluate.py \
    --config config/yolo_detection.yaml \
    --model "$DET_WEIGHTS" \
    --run-name "eval-det-${RUN_NAME}" \
    2>&1 | tee "logs/eval_det_${RUN_NAME}.log"

python visualize.py \
    --config config/yolo_detection.yaml \
    --model "$DET_WEIGHTS" \
    --n-samples 60 \
    --fps 3 \
    --run-name "vis-det-${RUN_NAME}" \
    2>&1 | tee "logs/vis_det_${RUN_NAME}.log"

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
echo ""
echo "============================================================"
echo "  Pipeline complete"
echo "============================================================"
echo "  Segmentation weights : $SEG_WEIGHTS"
echo "  Detection weights    : $DET_WEIGHTS"
echo "  MLflow results       : https://mlflow-geoai.stelarea.com/"
echo "  Logs                 : logs/"
echo ""
echo "  Next: deploy weights to backend/app/core/config.py"
echo "    model_segmentation_weights = \"$SEG_WEIGHTS\""
echo "    model_detection_weights    = \"$DET_WEIGHTS\""
