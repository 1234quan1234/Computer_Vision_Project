import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.clip_senet import ClipSENet
from loss.losses import LabelSmoothingCrossEntropy, SupConLoss


def test_sanity_overfit() -> None:
    batch_size, num_classes = 4, 776
    imgs = torch.randn(batch_size, 3, 320, 320)
    targets = torch.tensor([0, 0, 1, 1])

    # TensorDataset only accepts tensors; use a dummy placeholder for the path field.
    dummy_paths = torch.zeros(batch_size, dtype=torch.int64)
    dataset = TensorDataset(imgs, targets, targets, targets, dummy_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    use_sem = os.environ.get("ENABLE_CLIP", "0") == "1"
    model = ClipSENet(num_classes=num_classes, pretrained=False, use_sem=use_sem, use_afem=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce_fn = LabelSmoothingCrossEntropy(0.1)
    sc_fn = SupConLoss(0.07)

    model.train()
    for i, batch in enumerate(loader):
        imgs_batch, targets_batch, *_ = batch
        initial_loss = None
        acc = torch.tensor(0.0)

        for step in range(50):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs_batch, return_logits=True)
            feats = outputs["fusion_feat"]
            logits = outputs["logits"]
            loss = ce_fn(logits, targets_batch) + sc_fn(feats, targets_batch)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                acc = (logits.argmax(1) == targets_batch).float().mean()
                print(f"Iter {step:02d} | Loss: {loss.item():.4f} | Acc: {acc.item():.2f}")

        assert acc.item() == 1.0, f"Sanity check failed: acc={acc.item():.2f}"
        assert loss.item() < (initial_loss * 0.2), (
            f"Sanity check failed: loss did not drop enough "
            f"({initial_loss:.4f} -> {loss.item():.4f})"
        )
        break

    print("OK: sanity overfit")


if __name__ == "__main__":
    test_sanity_overfit()
