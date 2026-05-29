import os
import torch

from models.clip_senet import ClipSENet


def test_bf16_compatibility() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("SKIP: BF16 not supported on this device")
        return

    use_sem = os.environ.get("SKIP_CLIP", "0") != "1"
    model = ClipSENet(num_classes=776, pretrained=False, use_sem=use_sem, use_afem=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)

    dummy = torch.randn(2, 3, 320, 320, device="cuda")
    targets = torch.randint(0, 776, (2,), device="cuda")

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(dummy, return_logits=True)
        logits = outputs["logits"]
        loss = torch.nn.CrossEntropyLoss()(logits.float(), targets)

    loss.backward()
    assert not torch.isnan(loss), "BF16 loss is NaN"
    print("OK: BF16 autocast")


if __name__ == "__main__":
    test_bf16_compatibility()
