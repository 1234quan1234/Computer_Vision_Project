import random
from collections import defaultdict
from typing import Iterable, List, Tuple

from torch.utils.data import Sampler


class SpatioTemporalPKSampler(Sampler):
    """PK sampler with light spatio-temporal diversity heuristics."""

    def __init__(
        self,
        data_source,
        num_pids: int = 16,
        num_instances: int = 8,
        time_thresh: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.num_pids = num_pids
        self.num_instances = num_instances
        self.time_thresh = time_thresh
        self.seed = seed
        self.epoch = 0

        self.index_dic = defaultdict(list)
        for index, sample in enumerate(data_source.samples):
            _, pid, camid, frame = sample
            if pid == -1:
                continue
            self.index_dic[pid].append((index, camid, frame))

        self.pids = list(self.index_dic.keys())
        self.length = (len(self.pids) // self.num_pids) * self.num_pids * self.num_instances

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterable[int]:
        rng = random.Random(self.seed + self.epoch)
        pid_list = self.pids.copy()
        rng.shuffle(pid_list)

        batch_indices: List[int] = []
        final_indices: List[int] = []

        usable = len(pid_list) - len(pid_list) % self.num_pids
        for start in range(0, usable, self.num_pids):
            selected_pids = pid_list[start : start + self.num_pids]
            for pid in selected_pids:
                entries = self.index_dic[pid]
                batch_indices.extend(self._sample_instances(entries, rng))
            final_indices.extend(batch_indices)
            batch_indices = []

        return iter(final_indices)

    def __len__(self) -> int:
        return self.length

    def _sample_instances(self, entries: List[Tuple[int, int, int]], rng: random.Random) -> List[int]:
        if len(entries) >= self.num_instances and self.time_thresh > 0:
            entries_copy = entries.copy()
            rng.shuffle(entries_copy)
            selected: List[Tuple[int, int, int]] = []

            for idx, camid, frame in entries_copy:
                conflict = False
                for _, sel_cam, sel_frame in selected:
                    if camid == sel_cam and abs(frame - sel_frame) < self.time_thresh:
                        conflict = True
                        break
                if conflict:
                    continue
                selected.append((idx, camid, frame))
                if len(selected) >= self.num_instances:
                    break

            if len(selected) < self.num_instances:
                pool = entries_copy if entries_copy else entries
                while len(selected) < self.num_instances:
                    selected.append(rng.choice(pool))

            return [idx for idx, _, _ in selected]

        if len(entries) >= self.num_instances:
            cam_groups = defaultdict(list)
            for idx, camid, frame in entries:
                cam_groups[camid].append((frame, idx))

            for camid in cam_groups:
                cam_groups[camid].sort()

            cams = list(cam_groups.keys())
            rng.shuffle(cams)
            selected: List[int] = []

            # Round-robin across cameras, taking temporal extremes first.
            while len(selected) < self.num_instances and cams:
                next_cams = []
                for cam in cams:
                    group = cam_groups[cam]
                    if not group:
                        continue
                    take_front = (len(selected) % 2 == 0)
                    frame, idx = group.pop(0 if take_front else -1)
                    selected.append(idx)
                    if group:
                        next_cams.append(cam)
                    if len(selected) >= self.num_instances:
                        break
                cams = next_cams

            if len(selected) < self.num_instances:
                remaining = [idx for idx, _, _ in entries if idx not in selected]
                if not remaining:
                    remaining = [idx for idx, _, _ in entries]
                while len(selected) < self.num_instances:
                    selected.append(rng.choice(remaining))

            return selected

        pool = [idx for idx, _, _ in entries]
        return [rng.choice(pool) for _ in range(self.num_instances)]
