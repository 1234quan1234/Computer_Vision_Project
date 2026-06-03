from typing import Dict

import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_ce,
    loss_supcon,
    device,
    epoch: int,
    supcon_weight: float = 1.0,
    log_interval: int = 50,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    """Run one full training epoch over the dataset.

    Computes the combined loss:
        ``total_loss = CE_loss + supcon_weight * SupCon_loss``

    Uses BF16 autocast on CUDA for efficient mixed-precision training
    while keeping loss computation in FP32 for numerical stability.

    Args:
        model: The ``ClipSENet`` model to train.
        dataloader: Training DataLoader with PK-sampled batches.
        optimizer: AdamW optimizer with per-module learning rates.
        loss_ce: Label-smoothing Cross-Entropy loss function.
        loss_supcon: Supervised Contrastive loss function.
        device: Torch device (cuda or cpu).
        epoch: Current epoch number (for progress bar display).
        supcon_weight: Scalar multiplier for the SupCon loss term.
            Set to 0.0 to disable metric learning (CE-only ablation).
        log_interval: Print running averages every N steps.
        grad_clip: Maximum gradient norm for clipping. Set to 0.0 to disable.

    Returns:
        Dict with averaged training metrics: ``'loss'``, ``'loss_ce'``,
        and ``'loss_supcon'``.
    """
    model.train()
    autocast_enabled = device.type == "cuda"

    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images, pids, _, _, _ = batch
        images = images.to(device, non_blocking=True)
        targets = pids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = model(images, return_logits=True)
            logits = outputs["logits"]
            feats = outputs["fusion_feat"]

            # Keep loss math in FP32 while model/optimizer stay in BF16.
            loss_ce_val = loss_ce(logits.float(), targets)
            if supcon_weight > 0.0:
                loss_supcon_val = loss_supcon(feats.float(), targets)
            else:
                loss_supcon_val = torch.zeros((), device=device)

            loss = loss_ce_val + supcon_weight * loss_supcon_val

        loss.backward()
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_ce_val.item()
        total_supcon += loss_supcon_val.item()

        if log_interval and step % log_interval == 0:
            progress.set_postfix(
                loss=total_loss / step,
                loss_ce=total_ce / step,
                loss_supcon=total_supcon / step,
            )

    num_steps = max(1, len(dataloader))
    return {
        "loss": total_loss / num_steps,
        "loss_ce": total_ce / num_steps,
        "loss_supcon": total_supcon / num_steps,
    }
