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

for cfg in "${CONFIGS[@]}"; do
  base_out=$(python - "$cfg" <<'PY'
import os, sys, yaml
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
cfg = load_config(sys.argv[1])
print(cfg.get("output", {}).get("dir", "outputs/default"))
PY
)
  for seed in "${SEEDS[@]}"; do
    ckpt="${base_out}/seed_${seed}/best.pth"
    out_vis="outputs/visuals/$(basename "${base_out}")/seed_${seed}"
    python scripts/extract_visuals.py --config "$cfg" --checkpoint "$ckpt" --outputs-dir "$out_vis"
    python scripts/generate_gradcam.py --config "$cfg" --checkpoint "$ckpt" --output-dir "${out_vis}/gradcam"
  done
done