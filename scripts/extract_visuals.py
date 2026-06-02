#!/usr/bin/env python
import argparse
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from data.dataset import VeRi776Dataset
from data.transforms import build_val_transforms
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


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = True) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)


def extract_features(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    autocast_enabled = device.type == "cuda"

    feats = []
    pids = []
    camids = []
    paths = []

    with torch.no_grad():
        for images, pid, camid, _, path in tqdm(dataloader, desc="Extract", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(images, return_logits=False)
                feat = outputs["fusion_feat"]

            feat = F.normalize(feat.float(), dim=1).cpu().numpy()
            feats.append(feat)
            pids.extend(pid.cpu().numpy().tolist())
            camids.extend(camid.cpu().numpy().tolist())
            paths.extend(path)

    if feats:
        feats = np.vstack(feats)
    else:
        feats = np.zeros((0, 512), dtype=np.float32)

    return feats, np.asarray(pids), np.asarray(camids), paths


def compute_distance_matrix(qf: np.ndarray, gf: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if metric == "cosine":
        qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
        gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)
        return 1.0 - np.dot(qf, gf.T)

    qf_sq = np.sum(qf ** 2, axis=1, keepdims=True)
    gf_sq = np.sum(gf ** 2, axis=1, keepdims=True).T
    distmat = qf_sq + gf_sq - 2.0 * np.dot(qf, gf.T)
    return np.maximum(distmat, 0.0)


def apply_reid_mask(distmat, q_pids, g_pids, q_camids, g_camids) -> None:
    for q_idx in range(distmat.shape[0]):
        same = (g_pids == q_pids[q_idx]) & (g_camids == q_camids[q_idx])
        if np.any(same):
            distmat[q_idx, same] = np.inf


def first_match_rank(order: np.ndarray, q_pid: int, g_pids: np.ndarray) -> Optional[int]:
    matches = g_pids[order] == q_pid
    if not np.any(matches):
        return None
    return int(np.where(matches)[0][0]) + 1


def tensor_to_pil(img_tensor: torch.Tensor, mean=None, std=None) -> Image.Image:
    img = img_tensor.detach().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img)


def load_viz_image(path: str, val_transform) -> Image.Image:
    img = Image.open(path).convert("RGB")
    tensor = val_transform(img)
    return tensor_to_pil(tensor)


def add_border(img: Image.Image, color: str, border: int) -> Image.Image:
    return ImageOps.expand(img, border=border, fill=color)


def save_retrieval_grid(query_img: Image.Image, gallery_imgs: List[Image.Image], save_path: str, title: str):
    num_cols = 1 + len(gallery_imgs)
    fig, axes = plt.subplots(1, num_cols, figsize=(2.2 * num_cols, 2.8), dpi=200)
    if num_cols == 1:
        axes = [axes]

    axes[0].imshow(query_img)
    axes[0].axis("off")
    axes[0].set_title("Query")

    for idx, img in enumerate(gallery_imgs, start=1):
        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(f"#{idx}")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def sample_indices(indices: List[int], count: int, rng: random.Random) -> List[int]:
    if len(indices) <= count:
        return indices
    return rng.sample(indices, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features and generate retrieval visuals")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, default="outputs/visuals")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    num_workers = cfg["data"]["num_workers"]
    eval_bs = cfg["data"].get("eval_batch_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            device = torch.device("cpu")

    torch.backends.cudnn.benchmark = True

    val_transform = build_val_transforms(img_size=img_size)
    train_set = VeRi776Dataset(root=data_root, split="train", transform=val_transform)
    query_set = VeRi776Dataset(root=data_root, split="query", transform=val_transform)
    gallery_set = VeRi776Dataset(root=data_root, split="gallery", transform=val_transform)

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

    model = build_model(cfg, train_set.num_classes, device)
    load_checkpoint(model, args.checkpoint, strict=args.strict)

    qf, q_pids, q_camids, q_paths = extract_features(model, query_loader, device)
    gf, g_pids, g_camids, g_paths = extract_features(model, gallery_loader, device)

    visuals_dir = args.outputs_dir
    if not os.path.isabs(visuals_dir):
        visuals_dir = os.path.join(repo_root, visuals_dir)
    os.makedirs(visuals_dir, exist_ok=True)

    tsne_features = np.vstack([qf, gf])
    tsne_labels = np.concatenate([q_pids, g_pids])
    np.save(os.path.join(visuals_dir, "tsne_features.npy"), tsne_features)
    np.save(os.path.join(visuals_dir, "tsne_labels.npy"), tsne_labels)

    distmat = compute_distance_matrix(qf, gf, metric=args.metric)
    apply_reid_mask(distmat, q_pids, g_pids, q_camids, g_camids)
    order = np.argsort(distmat, axis=1)

    success = []
    near_miss = []
    hard_fail = []

    for q_idx in range(order.shape[0]):
        rank = first_match_rank(order[q_idx], int(q_pids[q_idx]), g_pids)
        if rank == 1:
            success.append(q_idx)
        elif rank in (2, 3):
            near_miss.append(q_idx)
        elif rank is None or rank > args.topk:
            hard_fail.append(q_idx)

    rng = random.Random(args.seed)
    success_samples = sample_indices(success, args.samples, rng)
    near_samples = sample_indices(near_miss, args.samples, rng)
    hard_samples = sample_indices(hard_fail, args.samples, rng)

    retrieval_dir = os.path.join(visuals_dir, "retrieval")
    near_dir = os.path.join(visuals_dir, "near_miss")
    hard_dir = os.path.join(visuals_dir, "hard_fail")
    os.makedirs(retrieval_dir, exist_ok=True)
    os.makedirs(near_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)

    def save_case(q_idx: int, out_dir: str, tag: str) -> None:
        q_pid = int(q_pids[q_idx])
        q_cam = int(q_camids[q_idx])
        q_img = load_viz_image(q_paths[q_idx], val_transform)
        q_img = add_border(q_img, "black", border=8)

        topk_idx = order[q_idx][: args.topk]
        gallery_imgs = []
        for g_idx in topk_idx:
            g_pid = int(g_pids[g_idx])
            g_img = load_viz_image(g_paths[g_idx], val_transform)
            color = "green" if g_pid == q_pid else "red"
            gallery_imgs.append(add_border(g_img, color, border=6))

        title = f"{tag} | q_pid={q_pid} q_cam={q_cam}"
        save_name = f"q{q_idx:04d}_pid{q_pid}.jpg"
        save_retrieval_grid(q_img, gallery_imgs, os.path.join(out_dir, save_name), title)

    for q_idx in tqdm(success_samples, desc="Save success"):
        save_case(q_idx, retrieval_dir, "success")

    for q_idx in tqdm(near_samples, desc="Save near_miss"):
        save_case(q_idx, near_dir, "near_miss")

    for q_idx in tqdm(hard_samples, desc="Save hard_fail"):
        save_case(q_idx, hard_dir, "hard_fail")

    print(f"Saved visuals to {os.path.relpath(visuals_dir, repo_root)}")


if __name__ == "__main__":
    main()
