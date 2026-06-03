import os
import re
from glob import glob
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _parse_filename(path: str) -> Tuple[int, int, int]:
    """Parse a VeRi-776 image filename to extract vehicle metadata.

    VeRi-776 filenames follow the pattern ``<pid>_c<camid>_<hhmmssff>.jpg``
    where *pid* is the vehicle identity, *camid* is the camera index, and
    the 8-digit suffix encodes hours/minutes/seconds/frames.

    Args:
        path: Absolute or relative path to an image file.

    Returns:
        A tuple of ``(pid, camid, timestamp)`` where *timestamp* is
        converted to a global frame count (30 fps assumed).
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # Expected pattern: 0001_c1_00030600 (pid_cam_hhmmssff)
    parts = name.split("_")
    pid = int(parts[0])
    camid = 0
    timestamp = 0

    # Extract camera id from the second token (e.g. "c1" -> 1).
    cam_token = parts[1] if len(parts) > 1 else ""
    cam_match = re.search(r"[cC](\d+)", cam_token) or re.search(r"[cC](\d+)", name)
    if cam_match:
        camid = int(cam_match.group(1))

    # Extract the timestamp from the last purely-numeric token.
    ts_token = ""
    for token in reversed(parts):
        if token.isdigit():
            ts_token = token
            break

    # Convert 8-digit timestamp (HHMMSSff) to a global frame index.
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
    """VeRi-776 image dataset with train/query/gallery splits.

    Each sample is a tuple of ``(image, pid, camid, frame, path)``.
    For the *train* split, ``pid`` values are remapped to contiguous
    integer labels ``[0, num_classes)`` suitable for Cross-Entropy loss.
    For *query* and *gallery*, original pids are preserved.

    Args:
        root: Path to the VeRi-776 dataset root directory containing
            ``image_train/``, ``image_query/``, and ``image_test/``.
        split: One of ``'train'``, ``'query'``, or ``'gallery'``.
        transform: Optional torchvision transform applied to each image.
    """

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
        """Scan the split directory and parse all image filenames.

        Returns:
            List of ``(path, pid, camid, frame)`` tuples sorted by path.
        """
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
        """Load and return a single sample.

        Args:
            index: Dataset index.

        Returns:
            Tuple of ``(image_tensor, pid, camid, frame, path)``.
            For the train split, *pid* is the remapped contiguous label.
        """
        path, pid, camid, frame = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.pid2label is not None:
            pid = self.pid2label[pid]

        return img, pid, camid, frame, path


def build_veri_datasets(root: str, train_transform, val_transform):
    """Convenience factory to build all three VeRi-776 dataset splits.

    Args:
        root: Path to VeRi-776 root (containing image_train/, image_query/, image_test/).
        train_transform: Augmentation pipeline for the training split.
        val_transform: Preprocessing pipeline for query and gallery splits.

    Returns:
        A tuple of ``(train_set, query_set, gallery_set)``.
    """
    train_set = VeRi776Dataset(root=root, split="train", transform=train_transform)
    query_set = VeRi776Dataset(root=root, split="query", transform=val_transform)
    gallery_set = VeRi776Dataset(root=root, split="gallery", transform=val_transform)
    return train_set, query_set, gallery_set
