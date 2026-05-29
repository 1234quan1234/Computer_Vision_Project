import torch

from models.sem import interpolate_pos_encoding

try:
    import clip
except ImportError:
    clip = None


def test_sem_interpolation() -> None:
    if clip is None:
        print("SKIP: CLIP is not installed")
        return

    model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
    new_pe = interpolate_pos_encoding(model, img_size=320, patch_size=16)

    assert new_pe.shape == (1, 401, 768), f"Interpolate shape mismatch: {new_pe.shape}"
    assert not torch.isnan(new_pe).any(), "Interpolate produced NaN"
    print("OK: SEM positional interpolation")


if __name__ == "__main__":
    test_sem_interpolation()
