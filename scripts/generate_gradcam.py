#!/usr/bin/env python
import argparse
import math
import os
import random
import sys
from typing import List

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from data.dataset import VeRi776Dataset
from data.transforms import build_val_transforms
from models.clip_senet import ClipSENet


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model: ClipSENet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x, return_logits=True)
        return outputs["logits"]


class UnNormalize:
    def __init__(self, mean, std) -> None:
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        out = tensor * std + mean
        return out.squeeze(0).clamp(0.0, 1.0)


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

    if strict:
        model.load_state_dict(state, strict=True)
        return

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []

    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value

    incompatible = model.load_state_dict(filtered_state, strict=False)

    if skipped_shape:
        print(f"[warn] Skipped {len(skipped_shape)} keys due to shape mismatch")
    if incompatible.missing_keys:
        print(f"[warn] Missing keys after load: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[warn] Unexpected keys after load: {len(incompatible.unexpected_keys)}")


def build_vit_reshape_transform(expected_tokens: int):
    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

        if tensor.shape[0] == expected_tokens:
            tensor = tensor.permute(1, 0, 2)
        elif tensor.shape[1] != expected_tokens:
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.permute(1, 0, 2)
            else:
                raise ValueError(f"Unexpected token count: {tuple(tensor.shape)}")

        tokens = tensor[:, 1:, :]
        batch, num_tokens, channels = tokens.shape
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens:
            raise ValueError(f"Token count {num_tokens} is not a square")

        tokens = tokens.reshape(batch, grid, grid, channels)
        return tokens.permute(0, 3, 1, 2)

    return reshape_transform


def to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def save_side_by_side(left: np.ndarray, right: np.ndarray, save_path: str) -> None:
    left_img = Image.fromarray(left)
    right_img = Image.fromarray(right)
    width = left_img.width + right_img.width
    height = max(left_img.height, right_img.height)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width, 0))
    canvas.save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for ResNet and CLIP branches")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/visuals/gradcam")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
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
    batch_size = args.batch_size or min(eval_bs, 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            device = torch.device("cpu")

    torch.backends.cudnn.benchmark = True

    val_transform = build_val_transforms(img_size=img_size)
    train_set = VeRi776Dataset(root=data_root, split="train", transform=val_transform)
    val_set = VeRi776Dataset(root=data_root, split="query", transform=val_transform)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_model(cfg, train_set.num_classes, device)
    load_checkpoint(model, args.checkpoint, strict=args.strict)
    model.eval()

    wrapper = LogitsWrapper(model)
    wrapper.eval()

    resnet_target = model.backbone.layer4[-1]
    vit_target = model.sem.visual.transformer.resblocks[-1]

    grid = img_size // cfg["model"].get("clip_patch_size", 16)
    expected_tokens = grid * grid + 1
    reshape_transform = build_vit_reshape_transform(expected_tokens)

    resnet_cam = GradCAM(model=wrapper, target_layers=[resnet_target])
    vit_cam = GradCAM(
        model=wrapper,
        target_layers=[vit_target],
        reshape_transform=reshape_transform,
    )

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    unnormalize = UnNormalize(CLIP_MEAN, CLIP_STD)
    mean_t = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    saved = 0
    with torch.set_grad_enabled(True):
        for images, _, _, _, paths in tqdm(val_loader, desc="Grad-CAM", leave=False):
            images = images.to(device, non_blocking=True)
            images.requires_grad_(True)

            resnet_grayscale = resnet_cam(input_tensor=images)
            vit_grayscale = vit_cam(input_tensor=images)

            clip_norm = (images - mean_t) / std_t
            vis_batch = unnormalize(clip_norm)

            for idx in range(images.shape[0]):
                if saved >= args.samples:
                    break

                rgb = to_rgb_numpy(vis_batch[idx])
                resnet_img = show_cam_on_image(rgb, resnet_grayscale[idx], use_rgb=True)
                vit_img = show_cam_on_image(rgb, vit_grayscale[idx], use_rgb=True)

                file_name = os.path.basename(paths[idx])
                save_path = os.path.join(output_dir, f"gradcam_{saved:02d}_{file_name}")
                save_side_by_side(resnet_img, vit_img, save_path)
                saved += 1

            if saved >= args.samples:
                break

    print(f"Saved {saved} Grad-CAM images to {os.path.relpath(output_dir, repo_root)}")


if __name__ == "__main__":
    main()
