import os
import re
from glob import glob
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _parse_filename(path: str) -> Tuple[int, int, int]:
    name = os.path.splitext(os.path.basename(path))[0]
    # Expected pattern: 0001_c1_00030600 (pid_cam_hhmmssff)
    parts = name.split("_")
    pid = int(parts[0])
    camid = 0
    timestamp = 0

    cam_token = parts[1] if len(parts) > 1 else ""
    cam_match = re.search(r"[cC](\d+)", cam_token) or re.search(r"[cC](\d+)", name)
    if cam_match:
        camid = int(cam_match.group(1))

    ts_token = ""
    for token in reversed(parts):
        if token.isdigit():
            ts_token = token
            break

    if ts_token and len(ts_token) == 8:
        hours = int(ts_token[0:2])
        minutes = int(ts_token[2:4])
        seconds = int(ts_token[4:6])
        frames = int(ts_token[6:8])
        timestamp = (hours * 3600 + minutes * 60 + seconds) * 30 + frames
    elif ts_token:
        timestamp = int(ts_token)

    return pid, camid, timestamp


class VeRi776Dataset(Dataset):
    """VeRi-776 image dataset with train/query/gallery splits."""

    def __init__(self, root: str, split: str = "train", transform=None) -> None:
        if split not in ("train", "query", "gallery"):
            raise ValueError(f"Unsupported split: {split}")
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = self._load_samples()
        self.pids = sorted({pid for _, pid, _, _ in self.samples if pid != -1})
        self.num_classes = len(self.pids)
        self.pid2label = {pid: idx for idx, pid in enumerate(self.pids)} if split == "train" else None

    def _load_samples(self) -> List[Tuple[str, int, int, int]]:
        split_dir = {
            "train": "image_train",
            "query": "image_query",
            "gallery": "image_test",
        }[self.split]

        data_dir = os.path.join(self.root, split_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Missing split directory: {data_dir}")

        image_paths = []
        for ext in IMG_EXTS:
            image_paths.extend(glob(os.path.join(data_dir, f"*{ext}")))
        image_paths.sort()

        samples = []
        for path in image_paths:
            pid, camid, frame = _parse_filename(path)
            samples.append((path, pid, camid, frame))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, pid, camid, frame = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.pid2label is not None:
            pid = self.pid2label[pid]

        return img, pid, camid, frame, path


def build_veri_datasets(root: str, train_transform, val_transform):
    train_set = VeRi776Dataset(root=root, split="train", transform=train_transform)
    query_set = VeRi776Dataset(root=root, split="query", transform=val_transform)
    gallery_set = VeRi776Dataset(root=root, split="gallery", transform=val_transform)
    return train_set, query_set, gallery_set
