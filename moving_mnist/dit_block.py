import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Multi-Head Self-Attention
# -------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (hidden_size // num_heads) ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Tokens, Hidden
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, Heads, N, dim]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# -------------------------
# MLP (Feed Forward Network)
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
# DiT Block (γ/β/α 调制版)
# -------------------------
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, mlp_ratio)

        # LayerNorm
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # conditioning projections
        # γ/β/α for attention + γ/β/α for MLP
        self.gamma1 = nn.Linear(hidden_size, hidden_size)
        self.beta1  = nn.Linear(hidden_size, hidden_size)
        self.alpha1 = nn.Linear(hidden_size, hidden_size)

        self.gamma2 = nn.Linear(hidden_size, hidden_size)
        self.beta2  = nn.Linear(hidden_size, hidden_size)
        self.alpha2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, cond):
        """
        x: [B, N, D] tokens
        cond: [B, D] conditioning embedding (timestep + label)
        """
        # ---- 1. 条件投影 ----
        gamma1_val = self.gamma1(cond)  # [B, D]
        beta1_val  = self.beta1(cond)
        alpha1_val = self.alpha1(cond)

        gamma2_val = self.gamma2(cond)
        beta2_val  = self.beta2(cond)
        alpha2_val = self.alpha2(cond)

        # ---- 2. Attention Block ----
        y = self.ln1(x)                                  # LN
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)  # γ/β 调制
        y = self.attn(y)                                 # Self-Attn
        y = y * alpha1_val.unsqueeze(1)                  # α 缩放
        x = x + y                                        # 残差

        # ---- 3. MLP Block ----
        z = self.ln2(x)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        z = self.mlp(z)
        z = z * alpha2_val.unsqueeze(1)
        x = x + z

        return x

if __name__=='__main__':
    dit_block=DiTBlock( hidden_size=16, num_heads=4)
    
    x=torch.rand((5,49,16))
    cond=torch.rand((5,16))
    
    outputs=dit_block(x,cond)
    print(outputs.shape)