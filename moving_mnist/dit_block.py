import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Multi-Head Self-Attention
# -------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
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
# MLP (Feed Forward)
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
# DiT Block (γ/β/α 调制版，稳定门控)
# -------------------------
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, gate_scale=0.1):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
        self.mlp  = MLP(hidden_size, mlp_ratio)
        self.ln1  = nn.LayerNorm(hidden_size)
        self.ln2  = nn.LayerNorm(hidden_size)

        # 条件映射：γ/β/α for attn + γ/β/α for mlp
        self.gamma1 = nn.Linear(hidden_size, hidden_size)
        self.beta1  = nn.Linear(hidden_size, hidden_size)
        self.alpha1 = nn.Linear(hidden_size, hidden_size)

        self.gamma2 = nn.Linear(hidden_size, hidden_size)
        self.beta2  = nn.Linear(hidden_size, hidden_size)
        self.alpha2 = nn.Linear(hidden_size, hidden_size)

        # 小尺度门控，防止训练初期不稳定
        self.gate_scale = gate_scale

        # 合理初始化（让调制初期温和）
        nn.init.zeros_(self.gamma1.weight); nn.init.zeros_(self.gamma1.bias)
        nn.init.zeros_(self.beta1.weight);  nn.init.zeros_(self.beta1.bias)
        nn.init.zeros_(self.gamma2.weight); nn.init.zeros_(self.gamma2.bias)
        nn.init.zeros_(self.beta2.weight);  nn.init.zeros_(self.beta2.bias)
        nn.init.zeros_(self.alpha1.weight); nn.init.zeros_(self.alpha1.bias)
        nn.init.zeros_(self.alpha2.weight); nn.init.zeros_(self.alpha2.bias)

    def forward(self, x, cond):
        """
        x: [B, N, D]
        cond: [B, D]
        """
        # cond → γ/β/α
        gamma1 = self.gamma1(cond)  # [B, D]
        beta1  = self.beta1(cond)
        # 用 tanh + 小尺度作为残差门控，稳定训练
        alpha1 = torch.tanh(self.alpha1(cond)) * self.gate_scale

        gamma2 = self.gamma2(cond)
        beta2  = self.beta2(cond)
        alpha2 = torch.tanh(self.alpha2(cond)) * self.gate_scale

        # Attn
        y = self.ln1(x)
        y = y * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        y = self.attn(y)
        x = x + y * alpha1.unsqueeze(1)

        # MLP
        z = self.ln2(x)
        z = z * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        z = self.mlp(z)
        x = x + z * alpha2.unsqueeze(1)

        return x


if __name__=='__main__':
    dit_block=DiTBlock( hidden_size=16, num_heads=4)
    
    x=torch.rand((5,49,16))
    cond=torch.rand((5,16))
    
    outputs=dit_block(x,cond)
    print(outputs.shape)  # [5, 49, 16]