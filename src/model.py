import math

import torch
import torch.nn as nn

from . import config as C


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Text2Pose(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = C.SEQ_LEN,
        pose_dim: int = C.POSE_DIM,
        d_model: int = C.D_MODEL,
        n_head: int = C.N_HEAD,
        n_layer: int = C.N_LAYER,
        dim_ff: int = C.DIM_FF,
        dropout: float = C.DROPOUT,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.query = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 8)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.head = nn.Linear(d_model, pose_dim)

    def forward(self, word_ids: torch.Tensor):
        b = word_ids.size(0)
        memory = self.word_emb(word_ids).unsqueeze(1)
        tgt = self.pos_enc(self.query.expand(b, -1, -1))
        out = self.decoder(tgt=tgt, memory=memory)
        return self.head(out)


def pose_loss(pred: torch.Tensor, target: torch.Tensor, smooth_w: float = C.SMOOTH_WEIGHT):
    mse = torch.mean((pred - target) ** 2)
    if smooth_w > 0:
        d_pred = pred[:, 1:] - pred[:, :-1]
        d_tgt = target[:, 1:] - target[:, :-1]
        smooth = torch.mean((d_pred - d_tgt) ** 2)
        return mse + smooth_w * smooth, mse.item(), smooth.item()
    return mse, mse.item(), 0.0
