import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.clip_senet import ClipSENet
from loss.losses import LabelSmoothingCrossEntropy, SupConLoss


def test_sanity_overfit() -> None:
    batch_size, num_classes = 4, 776
    imgs = torch.randn(batch_size, 3, 320, 320)
    targets = torch.tensor([0, 0, 1, 1])

    dataset = TensorDataset(imgs, targets, targets, targets, ["dummy"] * batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    use_sem = os.environ.get("ENABLE_CLIP", "0") == "1"
    model = ClipSENet(num_classes=num_classes, pretrained=False, use_sem=use_sem, use_afem=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce_fn = LabelSmoothingCrossEntropy(0.1)
    sc_fn = SupConLoss(0.07)

    model.train()
    for i, batch in enumerate(loader):
        imgs_batch, targets_batch, *_ = batch
        for step in range(50):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs_batch, return_logits=True)
            feats = outputs["fusion_feat"]
            logits = outputs["logits"]
            loss = ce_fn(logits, targets_batch) + sc_fn(feats, targets_batch)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                acc = (logits.argmax(1) == targets_batch).float().mean()
                print(f"Iter {step:02d} | Loss: {loss.item():.4f} | Acc: {acc.item():.2f}")

        assert loss.item() < 0.5, f"Sanity check failed: loss={loss.item():.4f}"
        break

    print("OK: sanity overfit")


if __name__ == "__main__":
    test_sanity_overfit()
