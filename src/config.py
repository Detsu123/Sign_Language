from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
POSE_DIR = DATA_DIR / "pose_per_individual_videos"
META_FILE = DATA_DIR / "WLASL_v0.3.json"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"

NUM_WORDS = 10
MIN_VIDEOS_PER_WORD = 7
SEQ_LEN = 50
NUM_KEYPOINTS = 55
KP_DIM = 2
POSE_DIM = NUM_KEYPOINTS * KP_DIM

D_MODEL = 128
N_HEAD = 4
N_LAYER = 3
DIM_FF = 256
DROPOUT = 0.1

BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
SMOOTH_WEIGHT = 0.1
GRAD_CLIP = 1.0

DEVICE = "cuda"
SEED = 42
