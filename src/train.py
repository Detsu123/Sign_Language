import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import config as C
from .dataset import (
    WLASLPoseDataset,
    build_index,
    select_top_words,
    split_dataset,
)
from .model import Text2Pose, pose_loss


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders():
    top_words = select_top_words(C.META_FILE, C.NUM_WORDS, C.MIN_VIDEOS_PER_WORD)
    samples, word_to_id, id_to_word = build_index(C.POSE_DIR, top_words)
    print(f"Found {len(samples)} pose folders for {len(top_words)} words.")

    ds = WLASLPoseDataset(samples)
    print(f"Loaded {len(ds)} valid pose sequences.")

    train_idx, val_idx = split_dataset(ds)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if val_idx else None

    train_loader = DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = (
        DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    return train_loader, val_loader, word_to_id, id_to_word


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for word_ids, pose in loader:
            word_ids = word_ids.to(device)
            pose = pose.to(device)
            pred = model(word_ids)
            loss, _, _ = pose_loss(pred, pose)
            total += loss.item() * word_ids.size(0)
            n += word_ids.size(0)
    return total / max(n, 1)


def train(verbose: bool = True):
    set_seed()
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")

    train_loader, val_loader, word_to_id, id_to_word = build_dataloaders()
    vocab_size = len(word_to_id)

    model = Text2Pose(vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model parameters: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)

    history = {"train": [], "val": []}
    best_val = float("inf")
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{C.EPOCHS}", disable=not verbose)
        for word_ids, pose in pbar:
            word_ids = word_ids.to(device)
            pose = pose.to(device)

            pred = model(word_ids)
            loss, mse, smooth = pose_loss(pred, pose)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()

            running += loss.item() * word_ids.size(0)
            n += word_ids.size(0)
            pbar.set_postfix(loss=running / n, mse=mse, smooth=smooth)

        sched.step()
        train_loss = running / max(n, 1)
        history["train"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            history["val"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "word_to_id": word_to_id,
                        "id_to_word": id_to_word,
                        "epoch": epoch,
                    },
                    C.CKPT_DIR / "best.pt",
                )

        if verbose and epoch % 10 == 0:
            msg = f"Epoch {epoch}: train={train_loss:.5f}"
            if val_loss is not None:
                msg += f"  val={val_loss:.5f}  best={best_val:.5f}"
            print(msg)

    torch.save(
        {
            "model": model.state_dict(),
            "word_to_id": word_to_id,
            "id_to_word": id_to_word,
            "epoch": C.EPOCHS,
        },
        C.CKPT_DIR / "last.pt",
    )
    with open(C.CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f)
    return model, history, word_to_id, id_to_word
