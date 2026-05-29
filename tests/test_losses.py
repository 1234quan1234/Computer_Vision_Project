import torch

from loss.losses import LabelSmoothingCrossEntropy, SupConLoss


def test_losses() -> None:
    batch_size, num_classes, feat_dim = 8, 776, 512
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    feats = torch.randn(batch_size, feat_dim, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    ce_fn = LabelSmoothingCrossEntropy(eps=0.1)
    supcon_fn = SupConLoss(temperature=0.07)

    loss_ce = ce_fn(logits, targets)
    loss_sc = supcon_fn(feats, targets)
    total = loss_ce + loss_sc

    assert not torch.isnan(total) and not torch.isinf(total), "Loss is NaN or Inf"

    total.backward()
    assert logits.grad is not None and feats.grad is not None
    assert not torch.isnan(logits.grad).any() and not torch.isnan(feats.grad).any()
    print(f"OK: losses | CE={loss_ce.item():.4f} | SupCon={loss_sc.item():.4f}")


if __name__ == "__main__":
    test_losses()
