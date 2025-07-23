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
                                      nn.Linear(fc3_size, num_of_classes))
    
    def forward(self, antigen):
        batch_size = antigen.size(0)
        seq_len = antigen.size(1)
        #convert dim (N, L, esm_embedding) --> (N*L, esm_embedding)
        output = torch.reshape(antigen, (batch_size*seq_len, self.esm_embedding_size))
        output = self.ff_model(output)                                               
        return output


class BepiPredDDPM(nn.Module):
    def __init__(self, esm_embedding_size=1280, timestep_dim=64):
        super().__init__()
        # 时间步嵌入层
        self.timestep_embed = nn.Sequential(
            nn.Linear(timestep_dim, esm_embedding_size),
            nn.SiLU(),
            nn.Linear(esm_embedding_size, esm_embedding_size)
        )
        
        # 继承 BepiPred-3.0 的 FFNN
        self.ffnn = MyDenseNetWithSeqLen(esm_embedding_size=esm_embedding_size * 2)  # 输入拼接时间步信息
        self.classifier = nn.Linear(esm_embedding_size, 1)  # 二分类

    def forward(self, x, t):
        # x: 噪声化ESM嵌入 [B, L, D]
        # t: 时间步 [B]
        t_embed = self.timestep_embed(t.unsqueeze(-1))  # [B, D]
        t_embed = t_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, D]
        print(t_embed.shape, x.shape)
        x = torch.cat([x, t_embed], dim=-1)
        noise_pred = self.ffnn(x)  # 预测噪声 [B, L, D]
        
        # 若需联合表位分类
        epitope_prob = torch.sigmoid(self.classifier(x))  # [B, L, 1]
        return noise_pred, epitope_prob


def eval_model(model, dataloader, device="cuda"):
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x0, epitope_labels in dataloader:
            _, epitope_prob = model(x0.to(device), torch.zeros(x0.size(0)).to(device))
            probs.append(epitope_prob.cpu())
            labels.append(epitope_labels.cpu())
    
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)  # 更关注正类的指标
    return pr_auc

class ESM2Dataset(Dataset):
    def __init__(self, esm_encoding_dir):
        self.esm_encoding_dir = esm_encoding_dir
        self.esm_files = [f for f in os.listdir(esm_encoding_dir) if f.endswith('.pt')]
        self.esm_embeddings, self.epitope_labels = self.load_data()

    def load_data(self):
        esm_embeddings = []
        epitope_labels = []
        for esm_file in self.esm_files:
            data = torch.load(self.esm_encoding_dir / esm_file)
            esm_embedding = data['esm_representation']
            epitope_label = data['epitope']
            esm_embeddings.append(esm_embedding)
            epitope_labels.append(epitope_label)
        return esm_embeddings, epitope_labels

    def __len__(self):
        return len(self.esm_files)

    def __getitem__(self, idx):
        esm_embedding = self.esm_embeddings[idx]
        epitope_label = self.epitope_labels[idx]
        return esm_embedding, epitope_label





if __name__ == "__main__":
    # 加载数据集
    dataset = ESM2Dataset(esm_encoding_dir=Path("data/esm_encodings"))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型和优化器
    model = BepiPredDDPM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    train(model, dataloader, optimizer, epochs=10)
