import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C


def select_top_words(meta_file: Path, num_words: int, min_videos: int):
    with open(meta_file, "r") as f:
        meta = json.load(f)
    entries = []
    for entry in meta:
        gloss = entry["gloss"]
        instances = entry.get("instances", [])
        if len(instances) >= min_videos:
            entries.append((gloss, instances))
    entries.sort(key=lambda x: -len(x[1]))
    return entries[:num_words]


def load_pose_json(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    people = data.get("people", [])
    if not people:
        return None
    keypoints = people[0]
    if not isinstance(keypoints, dict):
        return None
    parts = []
    for key in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        kp = keypoints.get(key, [])
        if not kp:
            return None
        arr = np.array(kp, dtype=np.float32).reshape(-1, 3)
        parts.append(arr[:, :2])
    pose = np.concatenate(parts, axis=0)
    if pose.shape[0] < C.NUM_KEYPOINTS:
        return None
    return pose[: C.NUM_KEYPOINTS]


def load_video_pose(video_dir):
    frame_files = sorted(video_dir.glob("*_keypoints.json"))
    if not frame_files:
        frame_files = sorted(video_dir.glob("*.json"))
    if not frame_files:
        return None
    frames = []
    for fp in frame_files:
        kp = load_pose_json(fp)
        if kp is not None:
            frames.append(kp)
    if len(frames) < 4:
        return None
    return np.stack(frames, axis=0)


def normalize_pose(pose):
    neck_idx = 1
    l_shoulder, r_shoulder = 5, 2
    centered = pose - pose[:, neck_idx : neck_idx + 1, :]
    shoulder_width = np.linalg.norm(
        pose[:, l_shoulder, :] - pose[:, r_shoulder, :], axis=-1
    )
    valid = shoulder_width[shoulder_width > 1e-3]
    if len(valid) == 0:
        return None
    scale = np.median(valid)
    if scale < 1e-3 or not np.isfinite(scale):
        return None
    return centered / scale


def resample_sequence(seq, target_len):
    n = seq.shape[0]
    if n == target_len:
        return seq
    idx = np.linspace(0, n - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    w = (idx - lo)[:, None, None]
    return seq[lo] * (1 - w) + seq[hi] * w


def build_index(pose_dir, top_words):
    samples = []
    word_to_id = {}
    id_to_word = {}
    for wid, (gloss, instances) in enumerate(top_words):
        word_to_id[gloss] = wid
        id_to_word[wid] = gloss
        for inst in instances:
            vid_id = inst["video_id"]
            vdir = pose_dir / vid_id
            if vdir.is_dir():
                samples.append((wid, vdir))
    return samples, word_to_id, id_to_word


class WLASLPoseDataset(Dataset):
    def __init__(self, samples, seq_len=C.SEQ_LEN):
        self.seq_len = seq_len
        self.cache = []
        for wid, vdir in samples:
            pose = load_video_pose(vdir)
            if pose is None:
                continue
            pose = normalize_pose(pose)
            if pose is None:
                continue
            pose = resample_sequence(pose, seq_len)
            pose = pose.reshape(seq_len, -1).astype(np.float32)
            if not np.isfinite(pose).all():
                continue
            self.cache.append((wid, pose))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        wid, pose = self.cache[idx]
        return torch.tensor(wid, dtype=torch.long), torch.from_numpy(pose)


def split_dataset(ds, train_ratio=0.85, seed=C.SEED):
    rng = np.random.default_rng(seed)
    by_word = defaultdict(list)
    for i, (wid, _) in enumerate(ds.cache):
        by_word[wid].append(i)
    train_idx, val_idx = [], []
    for wid, indices in by_word.items():
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * train_ratio))
        train_idx.extend(indices[:cut])
        val_idx.extend(indices[cut:])
    return train_idx, val_idx
