import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# Multi-Head Self-Attention
# -------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# -------------------------
# MLP / Feed-Forward
# -------------------------
class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# -------------------------
# DiTBlock with γ/β/α gate
# -------------------------
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, gate_scale=0.1):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
        self.mlp  = MLP(hidden_size, mlp_ratio)
        self.ln1  = nn.LayerNorm(hidden_size)
        self.ln2  = nn.LayerNorm(hidden_size)

        # 条件映射 γ/β/α
        self.gamma1 = nn.Linear(hidden_size, hidden_size)
        self.beta1  = nn.Linear(hidden_size, hidden_size)
        self.alpha1 = nn.Linear(hidden_size, hidden_size)

        self.gamma2 = nn.Linear(hidden_size, hidden_size)
        self.beta2  = nn.Linear(hidden_size, hidden_size)
        self.alpha2 = nn.Linear(hidden_size, hidden_size)

        # 初始化为 0，让 residual gate 初期温和
        for layer in [self.gamma1, self.beta1, self.alpha1,
                      self.gamma2, self.beta2, self.alpha2]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.gate_scale = gate_scale

    def forward(self, x, cond):
        """
        x: [B, N, D]
        cond: [B, D]
        """
        # ----------------- Attention -----------------
        y = self.ln1(x)
        gamma1 = self.gamma1(cond).unsqueeze(1)
        beta1  = self.beta1(cond).unsqueeze(1)
        alpha1 = torch.tanh(self.alpha1(cond)).unsqueeze(1) * self.gate_scale

        y = y * (1 + gamma1) + beta1
        y = self.attn(y)
        x = x + y * alpha1  # residual gate

        # ----------------- MLP -----------------
        z = self.ln2(x)
        gamma2 = self.gamma2(cond).unsqueeze(1)
        beta2  = self.beta2(cond).unsqueeze(1)
        alpha2 = torch.tanh(self.alpha2(cond)).unsqueeze(1) * self.gate_scale

        z = z * (1 + gamma2) + beta2
        z = self.mlp(z)
        x = x + z * alpha2  # residual gate

        return x


if __name__=='__main__':
    dit_block=DiTBlock( hidden_size=16, num_heads=4)
    
    x=torch.rand((5,49,16))
    cond=torch.rand((5,16))
    
    outputs=dit_block(x,cond)
    print(outputs.shape)  # [5, 49, 16]