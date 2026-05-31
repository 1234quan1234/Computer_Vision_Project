import argparse
import logging
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from data.dataset import build_veri_datasets
from data.sampler import SpatioTemporalPKSampler
from data.transforms import build_train_transforms, build_val_transforms
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from loss.losses import LabelSmoothingCrossEntropy, SupConLoss
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, epoch, best_map):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_map": best_map,
    }
    torch.save(state, path)


def setup_logging(output_dir: str, config_path: str) -> logging.Logger:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Config: %s", os.path.abspath(config_path))
    logger.info("Output dir: %s", os.path.abspath(output_dir))
    logger.info("Log file: %s", os.path.abspath(log_path))
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train CLIP-SEM ReID model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if cfg["data"]["batch_size"] != cfg["data"]["p"] * cfg["data"]["k"]:
        raise ValueError("batch_size must equal P*K for the PK sampler")

    train_tf = build_train_transforms(
        img_size=cfg["data"]["img_size"],
        color_jitter=cfg["data"].get("color_jitter", True),
        random_erasing=cfg["data"].get("random_erasing", True),
    )
    val_tf = build_val_transforms(img_size=cfg["data"]["img_size"])

    train_set, query_set, gallery_set = build_veri_datasets(cfg["data"]["root"], train_tf, val_tf)

    sampler = SpatioTemporalPKSampler(
        train_set,
        num_pids=cfg["data"]["p"],
        num_instances=cfg["data"]["k"],
        time_thresh=cfg["data"].get("time_thresh", 0),
        seed=cfg.get("seed", 42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    eval_bs = cfg["data"].get("eval_batch_size", 256)
    query_loader = DataLoader(
        query_set,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = ClipSENet(
        num_classes=train_set.num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        use_sem=cfg["model"].get("use_sem", True),
        use_afem=cfg["model"].get("use_afem", True),
        afem_groups=cfg["model"].get("afem_groups", 32),
        clip_img_size=cfg["model"].get("clip_img_size", 320),
        clip_patch_size=cfg["model"].get("clip_patch_size", 16),
        unfreeze_last_blocks=cfg["model"].get("unfreeze_last_blocks", 4),
    )
    model = model.to(device=device)

    base_lr = cfg["optim"]["lr"]
    lr_backbone = cfg["optim"].get("lr_backbone", base_lr)
    lr_sem = cfg["optim"].get("lr_sem", base_lr)
    lr_head = cfg["optim"].get("lr_head", base_lr)
    lr_afem = cfg["optim"].get("lr_afem", lr_head)

    param_groups = []

    def add_group(params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr})

    add_group(model.backbone.parameters(), lr_backbone)
    if model.use_sem and model.sem is not None:
        add_group(model.sem.parameters(), lr_sem)
    if model.use_afem and model.afem is not None:
        add_group(model.afem.parameters(), lr_afem)

    head_params = (
        list(model.concat_fc.parameters())
        + list(model.concat_norm.parameters())
        + list(model.classifier.parameters())
    )
    add_group(head_params, lr_head)

    if param_groups:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["optim"]["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=cfg["optim"]["weight_decay"],
        )

    epochs = cfg["train"]["epochs"]
    warmup_epochs = int(cfg["optim"].get("warmup_epochs", 0))

    def warmup_cosine_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    loss_ce = LabelSmoothingCrossEntropy(eps=cfg["loss"]["label_smoothing"])
    loss_supcon = SupConLoss(temperature=cfg["loss"]["supcon_temp"])

    output_dir = args.output_dir or cfg["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, args.config)
    logger.info("Seed: %s", cfg.get("seed", 42))

    best_map = 0.0
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_ce=loss_ce,
            loss_supcon=loss_supcon,
            device=device,
            epoch=epoch,
            supcon_weight=cfg["loss"]["supcon_weight"],
            log_interval=cfg["train"].get("log_interval", 50),
            grad_clip=cfg["train"].get("grad_clip", 0.0),
        )

        if epoch % cfg["train"]["eval_period"] == 0 or epoch == epochs:
            eval_metrics = evaluate(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
                rerank=cfg["eval"].get("rerank", True),
                k1=cfg["eval"].get("k1", 20),
                k2=cfg["eval"].get("k2", 6),
                lambda_value=cfg["eval"].get("lambda_value", 0.3),
            )

            mAP = eval_metrics["mAP"]
            if mAP > best_map:
                best_map = mAP
                save_checkpoint(os.path.join(output_dir, "best.pth"), model, optimizer, epoch, best_map)

        save_checkpoint(os.path.join(output_dir, "last.pth"), model, optimizer, epoch, best_map)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"mAP={best_map:.4f}"
        )

        scheduler.step()


if __name__ == "__main__":
    main()
