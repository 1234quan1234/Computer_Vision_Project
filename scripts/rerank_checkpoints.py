import argparse
import csv
import fnmatch
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from data.dataset import build_veri_datasets
from data.transforms import build_train_transforms, build_val_transforms
from engine.evaluator import evaluate
from models.clip_senet import ClipSENet


def merge_dicts(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = cfg.get("base")
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_path))
        base_cfg = load_config(base_path)
        cfg = merge_dicts(base_cfg, {k: v for k, v in cfg.items() if k != "base"})

    return cfg


def normalize_output_dir(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(os.path.relpath(path, repo_root))
    return os.path.normpath(path)


def iter_yaml_paths(root_dir: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith((".yaml", ".yml")):
                paths.append(os.path.join(dirpath, name))
    return paths


def build_config_map(config_root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in iter_yaml_paths(config_root):
        try:
            cfg = load_config(path)
        except Exception:
            continue
        output_dir = cfg.get("output", {}).get("dir")
        if output_dir:
            mapping[normalize_output_dir(output_dir)] = path
    return mapping


def find_config_for_checkpoint(
    ckpt_path: str,
    outputs_root: str,
    config_map: Dict[str, str],
    default_config: str,
) -> str:
    outputs_root_abs = outputs_root
    if not os.path.isabs(outputs_root_abs):
        outputs_root_abs = os.path.join(repo_root, outputs_root_abs)

    rel = os.path.relpath(ckpt_path, outputs_root_abs)
    rel_parts = rel.split(os.sep)
    if rel_parts:
        output_dir_abs = os.path.join(outputs_root_abs, rel_parts[0])
        output_dir_key = normalize_output_dir(output_dir_abs)
        if output_dir_key in config_map:
            return config_map[output_dir_key]

    return default_config


def build_loaders(cfg) -> Tuple[int, DataLoader, DataLoader]:
    train_tf = build_train_transforms(
        img_size=cfg["data"]["img_size"],
        color_jitter=cfg["data"].get("color_jitter", True),
        random_erasing=cfg["data"].get("random_erasing", True),
    )
    val_tf = build_val_transforms(img_size=cfg["data"]["img_size"])

    train_set, query_set, gallery_set = build_veri_datasets(cfg["data"]["root"], train_tf, val_tf)

    eval_bs = cfg["data"].get("eval_batch_size", 256)
    num_workers = cfg["data"]["num_workers"]

    query_loader = DataLoader(
        query_set,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_set.num_classes, query_loader, gallery_loader


def list_checkpoints(outputs_root: str, patterns: List[str]) -> List[str]:
    outputs_root_abs = outputs_root
    if not os.path.isabs(outputs_root_abs):
        outputs_root_abs = os.path.join(repo_root, outputs_root_abs)

    matches: List[str] = []
    for dirpath, _, filenames in os.walk(outputs_root_abs):
        for name in filenames:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    matches.append(os.path.join(dirpath, name))
                    break

    matches.sort()
    return matches


def load_state(model, ckpt_path: str, strict: bool) -> bool:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        print(f"[skip] load failed for {ckpt_path}: {exc}")
        return False
    return True


def build_model(cfg, num_classes: int, device: torch.device) -> ClipSENet:
    model = ClipSENet(
        num_classes=num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        use_sem=cfg["model"].get("use_sem", True),
        use_afem=cfg["model"].get("use_afem", True),
        afem_groups=cfg["model"].get("afem_groups", 32),
        clip_img_size=cfg["model"].get("clip_img_size", 320),
        clip_patch_size=cfg["model"].get("clip_patch_size", 16),
        unfreeze_last_blocks=cfg["model"].get("unfreeze_last_blocks", 4),
    )
    return model.to(device=device)


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"mAP={metrics['mAP'] * 100:.2f} "
        f"R1={metrics['r1'] * 100:.2f} "
        f"R5={metrics['r5'] * 100:.2f} "
        f"R10={metrics['r10'] * 100:.2f}"
    )


def write_csv(rows: List[Dict[str, str]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with re-ranking")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--pattern", nargs="+", default=["best.pth"])
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            device = torch.device("cpu")

    config_map = build_config_map(os.path.join(repo_root, "configs"))
    ckpt_paths = list_checkpoints(args.outputs_root, args.pattern)
    if not ckpt_paths:
        print("No checkpoints found.")
        return

    loader_cache: Dict[str, Tuple[int, DataLoader, DataLoader]] = {}
    rows: List[Dict[str, str]] = []

    for ckpt_path in ckpt_paths:
        cfg_path = find_config_for_checkpoint(ckpt_path, args.outputs_root, config_map, args.config)
        cfg = load_config(cfg_path)

        data_key = "|".join(
            str(x)
            for x in (
                cfg["data"]["root"],
                cfg["data"]["img_size"],
                cfg["data"].get("eval_batch_size", 256),
                cfg["data"]["num_workers"],
            )
        )
        if data_key not in loader_cache:
            loader_cache[data_key] = build_loaders(cfg)
        num_classes, query_loader, gallery_loader = loader_cache[data_key]

        model = build_model(cfg, num_classes, device)
        if not load_state(model, ckpt_path, args.strict):
            continue

        eval_metrics = evaluate(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            rerank=args.rerank,
            k1=args.k1,
            k2=args.k2,
            lambda_value=args.lambda_value,
        )

        cmc = eval_metrics["cmc"]
        metrics = {
            "mAP": float(eval_metrics["mAP"]),
            "r1": float(cmc[0]) if len(cmc) > 0 else 0.0,
            "r5": float(cmc[4]) if len(cmc) > 4 else 0.0,
            "r10": float(cmc[9]) if len(cmc) > 9 else 0.0,
        }

        ckpt_rel = os.path.relpath(ckpt_path, repo_root)
        print(f"{ckpt_rel}: {format_metrics(metrics)}")
        rows.append(
            {
                "checkpoint": ckpt_rel,
                "mAP": f"{metrics['mAP']:.6f}",
                "r1": f"{metrics['r1']:.6f}",
                "r5": f"{metrics['r5']:.6f}",
                "r10": f"{metrics['r10']:.6f}",
                "config": os.path.relpath(cfg_path, repo_root),
            }
        )

    if args.csv and rows:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(repo_root, csv_path)
        write_csv(rows, csv_path)
        print(f"Saved results to {os.path.relpath(csv_path, repo_root)}")


if __name__ == "__main__":
    main()
