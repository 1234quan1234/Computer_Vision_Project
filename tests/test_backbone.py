import torch

from models.backbone import ResNet18Backbone


def test_backbone() -> None:
    model = ResNet18Backbone(pretrained=False)
    dummy = torch.randn(4, 3, 320, 320) + 5.0

    model.train()
    mean_before = model.bnneck.running_mean.clone()
    feat, bn_feat = model(dummy)

    assert feat.shape == (4, 512), f"Feature shape mismatch: {feat.shape}"
    assert bn_feat.shape == (4, 512), f"BN feature shape mismatch: {bn_feat.shape}"

    mean_after = model.bnneck.running_mean
    assert (mean_after - mean_before).abs().sum() > 0, "BNNeck running mean did not update"

    model.eval()
    with torch.no_grad():
        feat_eval, bn_feat_eval = model(dummy)
    assert feat_eval.shape == (4, 512)
    assert bn_feat_eval.shape == (4, 512)
    print("OK: backbone")


if __name__ == "__main__":
    test_backbone()
