import os

import torch
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import build_veri_datasets
from data.sampler import SpatioTemporalPKSampler
from data.transforms import build_train_transforms


def _resolve_veri_root() -> str:
    return os.environ.get("VERI_ROOT", "/workspace/VeRi-776/VeRi")


def test_data_pipeline() -> None:
    # 1. Test transforms
    tf = build_train_transforms(img_size=320)
    dummy = torch.randint(0, 256, (400, 600, 3), dtype=torch.uint8).numpy()
    img_pil = Image.fromarray(dummy)
    out = tf(img_pil)
    assert out.shape == (3, 320, 320), f"Transform shape mismatch: {out.shape}"
    print("OK: transforms")

    # 2. Test dataset and sampler if data is available
    veri_root = _resolve_veri_root()
    train_dir = os.path.join(veri_root, "image_train")
    if not os.path.isdir(train_dir):
        print(f"SKIP: dataset not found at {train_dir}")
        return

    train_set, _, _ = build_veri_datasets(veri_root, tf, tf)
    assert len(train_set) > 0, "Dataset is empty"

    img, pid, camid, frame, path = train_set[0]
    assert img.shape == (3, 320, 320)
    assert isinstance(pid, int) and isinstance(camid, int) and isinstance(frame, int)
    assert isinstance(path, str) and path
    print(f"OK: dataset | samples={len(train_set)} | pids={train_set.num_classes}")

    if train_set.num_classes < 4:
        print("SKIP: not enough identities for PK sampling")
        return

    sampler = SpatioTemporalPKSampler(train_set, num_pids=4, num_instances=2, seed=123)
    loader = DataLoader(
        train_set,
        batch_size=8,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    batch = next(iter(loader))
    imgs, pids, cams, frames, paths = batch
    assert imgs.shape == (8, 3, 320, 320)
    assert len(torch.unique(pids)) == 4, "Sampler does not enforce P identities"
    print("OK: sampler")


if __name__ == "__main__":
    test_data_pipeline()
