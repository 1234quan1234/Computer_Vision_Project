#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/default.yaml
  configs/ablation/abl_baseline.yaml
  configs/ablation/abl_no_afem.yaml
  configs/ablation/abl_no_supcon.yaml
  configs/ablation/abl_freeze_clip.yaml
  configs/ablation/abl_no_colorjitter.yaml
  configs/ablation/abl_low_lr.yaml
)

SEEDS=(42 3407 2023)

get_output_dir() {
  python - "$1" <<'PY'
import os
import sys
import yaml

path = sys.argv[1]

def load_config(p):
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.get("base")
    if base:
        if not os.path.isabs(base):
            base = os.path.normpath(os.path.join(os.path.dirname(p), base))
        base_cfg = load_config(base)
        base_cfg.update({k: v for k, v in cfg.items() if k != "base"})
        return base_cfg
    return cfg

cfg = load_config(path)
print(cfg.get("output", {}).get("dir", "outputs/default"))
PY
}

for seed in "${SEEDS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    base_out=$(get_output_dir "$cfg")
    output_dir="${base_out}/seed_${seed}"
    echo "Seed ${seed} | Config ${cfg} | Output ${output_dir}"
    python scripts/train.py --config "$cfg" --seed "$seed" --output-dir "$output_dir"
  done
done
