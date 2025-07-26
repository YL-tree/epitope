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
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence

 
class Diffusion:
    def __init__(self, device='cuda', timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        
        # Posterior variance calculation
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(device)

    def extract(self, a, t, x_shape):
        """从a中根据t提取系数并重塑使其能与x_shape广播"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：q(x_t | x_0)（公式4推导）"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, pred_noise, t, t_index):
        """反向扩散过程：p(x_{t-1} | x_t)（公式11）"""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # 论文中的公式11
        model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

      

class MyDenseNetWithSeqLen(nn.Module):
    def __init__(self,
                 esm_embedding_size = 1281,
                 fc1_size = 150,
                 fc2_size = 120,
                 fc3_size = 45,
                 fc1_dropout = 0.7,
                 fc2_dropout = 0.7,
                 fc3_dropout = 0.7,
                 num_of_classes = 2):
        super(MyDenseNetWithSeqLen, self).__init__()
        
        
        self.esm_embedding_size = esm_embedding_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout
        
        self.ff_model = nn.Sequential(nn.Linear(esm_embedding_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, esm_embedding_size))
    
    def forward(self, antigen):
        batch_size = antigen.size(0)
        seq_len = antigen.size(1)
        print(antigen.shape)
        #convert dim (N, L, esm_embedding) --> (N*L, esm_embedding)
        output = torch.reshape(antigen, (batch_size*seq_len, self.esm_embedding_size))
        output = self.ff_model(output)  
        #convert dim (N*L, esm_embedding) --> (N, L, esm_embedding)
        output = torch.reshape(output, (batch_size, seq_len, self.esm_embedding_size))

        return output



class BepiPredDDPM(nn.Module):
    def __init__(self, esm_embedding_size=1281, timestep_dim=64):
        super().__init__()
        # 时间步嵌入层
        self.timestep_embed = nn.Sequential(
            nn.Linear(timestep_dim, timestep_dim * 2),
            nn.SiLU(),
            nn.Linear(timestep_dim * 2, esm_embedding_size)
        )
        
        # 更强的特征提取主干（替换原FFNN）
        self.feature_extractor = nn.Sequential(
            nn.Linear(esm_embedding_size * 2, 1024),  # 输入拼接时间步
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        # 噪声预测头（扩散任务）
        self.noise_head = nn.Linear(512, esm_embedding_size)

        
        # 表位分类头（重点优化）
        self.epitope_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid())  # 输出概率
        

    def get_time_embedding(self, t, dim):
        # t: 时间步张量, shape=(batch_size, 1)
        # dim: 嵌入向量的维度
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        # emb.shape ==> (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # shape=(batch_size, dim)
        return emb

    def forward(self, x, t):
        # x: 噪声化ESM嵌入 [B, L, D]
        # t: 时间步 [B]
        t = t.unsqueeze(-1)
        t_embed = self.get_time_embedding(t, dim=64)  # [B, D]
        t_embed = self.timestep_embed(t_embed)
        t_embed = t_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, D]
        # print(t_embed.shape, x.shape)
        # x = t_embed + x
        x = torch.cat([x, t_embed], dim=-1)
        x = self.feature_extractor(x)
        noise_pred = self.noise_head(x)  # 预测噪声 [B, L, D]
        # 若需联合表位分类
        # print(x.shape)
        
        epitope_prob = self.epitope_head(x)  # [B, L, 1]
        return noise_pred, epitope_prob




if __name__ == "__main__":
    # 加载数据集
    # dataset = ESM2Dataset(esm_encoding_dir=Path("data/esm_encodings"))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型和优化器
    model = BepiPredDDPM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
