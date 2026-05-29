import os
import torch

from models.clip_senet import ClipSENet


def test_full_model() -> None:
    use_sem = os.environ.get("SKIP_CLIP", "0") != "1"

    try:
        model = ClipSENet(num_classes=776, pretrained=False, use_sem=use_sem, use_afem=True)
    except Exception as exc:
        print(f"WARN: CLIP init failed ({exc}). Retrying with use_sem=False")
        model = ClipSENet(num_classes=776, pretrained=False, use_sem=False, use_afem=True)

    dummy = torch.randn(2, 3, 320, 320)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy, return_logits=True)

    feats = outputs["fusion_feat"]
    logits = outputs["logits"]

    assert feats.shape == (2, 512), f"Fusion feature shape mismatch: {feats.shape}"
    assert logits.shape == (2, 776), f"Logits shape mismatch: {logits.shape}"
    print("OK: full model forward")


if __name__ == "__main__":
    test_full_model()
