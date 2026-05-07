from pathlib import Path

import torch

from . import config as C
from .model import Text2Pose


def load_model(ckpt_path: Path, device: str = None):
    device = device or (C.DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    word_to_id = ckpt["word_to_id"]
    id_to_word = ckpt["id_to_word"]

    model = Text2Pose(vocab_size=len(word_to_id)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, word_to_id, id_to_word, device


@torch.no_grad()
def generate(model, word: str, word_to_id: dict, device: str):
    if word not in word_to_id:
        raise KeyError(f"Unknown word '{word}'. Available: {list(word_to_id)}")
    wid = torch.tensor([word_to_id[word]], dtype=torch.long, device=device)
    pose = model(wid)[0].cpu().numpy()
    return pose
