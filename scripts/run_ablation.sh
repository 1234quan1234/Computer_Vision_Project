#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/ablation/abl_baseline.yaml
  configs/ablation/abl_no_afem.yaml
  configs/ablation/abl_no_supcon.yaml
  configs/ablation/abl_no_rerank.yaml
  configs/ablation/abl_freeze_clip.yaml
  configs/ablation/abl_no_colorjitter.yaml
  configs/ablation/abl_low_lr.yaml
)

for cfg in "${CONFIGS[@]}"; do
  python scripts/train.py --config "$cfg"
done
