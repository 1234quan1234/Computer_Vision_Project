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
) -> Dict[str, float]:
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
