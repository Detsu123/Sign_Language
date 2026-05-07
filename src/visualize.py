from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from . import config as C


BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8),
]

LEFT_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
RIGHT_HAND_EDGES = LEFT_HAND_EDGES

BODY_COUNT = 13
LH_OFFSET = BODY_COUNT
RH_OFFSET = BODY_COUNT + 21


def render_animation(pose_seq: np.ndarray, save_path: Path, fps: int = 12, title: str = ""):
    """pose_seq: (T, 55, 2) normalized coordinates"""
    fig, ax = plt.subplots(figsize=(4, 5))

    flat = pose_seq.reshape(-1, 2)
    pad = 0.5
    xmin, xmax = flat[:, 0].min() - pad, flat[:, 0].max() + pad
    ymin, ymax = flat[:, 1].min() - pad, flat[:, 1].max() + pad

    def draw_frame(t):
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  frame {t+1}/{len(pose_seq)}")
        ax.set_xticks([])
        ax.set_yticks([])

        kp = pose_seq[t]
        body = kp[:BODY_COUNT]
        lh = kp[LH_OFFSET : LH_OFFSET + 21]
        rh = kp[RH_OFFSET : RH_OFFSET + 21]

        for i, j in BODY_EDGES:
            if i < len(body) and j < len(body):
                ax.plot([body[i, 0], body[j, 0]], [body[i, 1], body[j, 1]], "b-", lw=2)
        ax.scatter(body[:, 0], body[:, 1], c="b", s=20, zorder=3)

        for i, j in LEFT_HAND_EDGES:
            ax.plot([lh[i, 0], lh[j, 0]], [lh[i, 1], lh[j, 1]], "g-", lw=1)
        ax.scatter(lh[:, 0], lh[:, 1], c="g", s=8, zorder=3)

        for i, j in RIGHT_HAND_EDGES:
            ax.plot([rh[i, 0], rh[j, 0]], [rh[i, 1], rh[j, 1]], "r-", lw=1)
        ax.scatter(rh[:, 0], rh[:, 1], c="r", s=8, zorder=3)

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(pose_seq), interval=1000 / fps
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix == ".gif":
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        anim.save(save_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return save_path


def pose_tensor_to_numpy(pose: np.ndarray):
    """(T, 110) -> (T, 55, 2)"""
    return pose.reshape(pose.shape[0], C.NUM_KEYPOINTS, C.KP_DIM)
