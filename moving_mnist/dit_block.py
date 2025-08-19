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
# Adaptive LayerNorm-Zero
# -------------------------
class AdaLNZero(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    
    def forward(self, x, shift, scale):
        return (1 + scale) * self.ln(x) + shift

# -------------------------
# DiT Block with adaLN-Zero
# -------------------------
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, mlp_ratio)
        
        # AdaLN-zero parameters α, β, γ
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_mlp  = nn.Parameter(torch.zeros(1))
        
        self.ln1 = AdaLNZero(hidden_size)
        self.ln2 = AdaLNZero(hidden_size)

        # Conditioning projection (maps timestep/label → shift/scale)
        self.condition_proj = nn.Linear(hidden_size, hidden_size * 4)

    def forward(self, x, cond):
        """
        x: [B, N, D] input tokens
        cond: [B, D] conditioning embedding (timestep + label)
        """
        # 1. get γ1, β1, γ2, β2
        cond_out = self.condition_proj(cond)  # [B, 4*D]
        shift_msa, scale_msa, shift_mlp, scale_mlp = cond_out.chunk(4, dim=-1)

        # 2. Attention block
        x = x + self.alpha_attn * self.attn(
            self.ln1(x, shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        )

        # 3. MLP block
        x = x + self.alpha_mlp * self.mlp(
            self.ln2(x, shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        )
        return x
