### IMPORTS ###
import math
import subprocess
import numpy as np
import torch
import esm
import csv
import torch.nn as nn
from pathlib import Path
import sys
import os
import pandas as pd

### SET GPU OR CPU ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU device detected: {device}")
else:
    device = torch.device("cpu")
    print(f"GPU device not detected. Using CPU: {device}")

def get_time_embedding(t, dim):
        # t: 时间步张量, shape=(batch_size, 1)
        # dim: 嵌入向量的维度
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        # emb.shape ==> (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # shape=(batch_size, dim)
        return emb

class DDPM_Predicet(nn.Module):
    def __init__(self, ESM_emb=1281, emb_dim=1024, num_layers=4, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim

        # Embedding layers
        self.seq_embedding = nn.Linear(ESM_emb, emb_dim)
        self.bepi_project = nn.Linear(1, emb_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(emb_dim, 1)  # Predict noise or mask value


    def forward(self, noisy_mask, seq, t):
        """
        noisy_mask: (B, L, 1) - optional, for conditioning or residual use
        seq:        (B, L, ESM_emb) - int token ids
        t:          (B,) - time step
        """
        B, L, D = seq.shape     
        # assert L <= self.seq_len

        # Embedding inputs
        seq_emb = self.seq_embedding(seq)                               # (B, L, emb_dim)
        noisy_mask_emb = self.bepi_project(noisy_mask)        # (B, L, emb_dim)
        time_emb = get_time_embedding(t.unsqueeze(-1), self.emb_dim)   # (B, emb_dim)
        time_emb = time_emb.unsqueeze(1).expand(B, L, -1)                   # (B, L, emb_dim)
        
        # print(seq_emb.shape, noisy_mask_emb.shape, time_emb.shape)
        # Combine all embeddings
        x = seq_emb + noisy_mask_emb + time_emb                             # (B, L, emb_dim)

        # Transformer encoding
        x = self.transformer(x)                                             # (B, L, emb_dim)

        # Project to 1D mask prediction
        out = self.output_proj(x).squeeze(-1)                               # (B, L)

        return out

class DDPM_enhance(nn.Module):
    def __init__(self, ESM_emb=1281, emb_dim=1024, num_layers=4, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim

        # Embedding layers
        self.seq_embedding = nn.Linear(ESM_emb, emb_dim)
        self.bepi_project = nn.Linear(1, emb_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(emb_dim, 1)  # Predict noise or mask value


    def forward(self, noisy_mask, seq, bepi_scores, t):
        """
        noisy_mask: (B, L, 1) - optional, for conditioning or residual use
        seq:        (B, L, ESM_emb)
        bepi_scores:(B, L) - float [0~1]
        t:          (B,) - time step
        """
        B, L, D = seq.shape
        # assert L <= self.seq_len

        # Embedding inputs
        seq_emb = self.seq_embedding(seq)                              # (B, L, emb_dim)
        noisy_mask_emb = self.bepi_project(noisy_mask)                 # (B, L, emb_dim)
        bepi_emb = self.bepi_project(bepi_scores.unsqueeze(-1))        # (B, L, emb_dim)
        time_emb = get_time_embedding(t.unsqueeze(-1), self.emb_dim)   # (B, emb_dim)
        time_emb = time_emb.unsqueeze(1).expand(B, L, -1)              # (B, L, emb_dim)
        
        # Combine all embeddings
        x = seq_emb + bepi_emb + noisy_mask_emb + time_emb                             # (B, L, emb_dim)

        # Optional: add residual noisy mask as conditioning
        # x = x + noisy_mask  # If you want to directly feed noisy mask values

        # Transformer encoding
        x = self.transformer(x)                                        # (B, L, emb_dim)

        # Project to 1D mask prediction
        out = self.output_proj(x).squeeze(-1)                          # (B, L)

        return out