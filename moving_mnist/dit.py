import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dit_block import DiTBlock  # 上面改好的 DiTBlock

# -----------------------
# Sinusoidal Time Embedding
# -----------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# -----------------------
# Patch Embedding / Unpatchify (完全对齐参考)
# -----------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        # patchify conv
        self.conv = nn.Conv2d(in_channels, in_channels * patch_size ** 2, 
                              kernel_size=patch_size, stride=patch_size)
        # token projection
        self.proj = nn.Linear(in_channels * patch_size ** 2, embed_dim)
        # learnable position embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.conv(x)  # [B, C*p*p, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C*p*p]
        x = self.proj(x)  # [B, N, embed_dim]
        x = x + self.pos_emb
        Hs = Ws = self.img_size // self.patch_size
        return x, (Hs, Ws)

class Unpatchify(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        # map token -> patch channels
        self.linear = nn.Linear(embed_dim, out_channels * patch_size ** 2)

    def forward(self, tokens, spatial_shape):
        B, N, D = tokens.shape
        Hs, Ws = spatial_shape
        x = self.linear(tokens)  # [B, N, C*p*p]
        x = x.view(B, Hs, Ws, -1)  # [B, Hs, Ws, C*p*p]
        x = x.permute(0, 3, 1, 2)  # [B, C*p*p, Hs, Ws]
        x = x.reshape(B, -1, Hs * self.patch_size, Ws * self.patch_size)  # [B, C, H, W]
        return x

# -----------------------
# DiT Network
# -----------------------
class DiT(nn.Module):
    def __init__(self,
                 image_size=28,
                 in_channels=1,
                 patch_size=4,
                 embed_dim=64,
                 depth=4,
                 num_heads=4,
                 mlp_ratio=4.0,
                 timestep_emb_dim=128,
                 num_classes=None,
                 out_channels=None
                 ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.out_channels = out_channels or in_channels

        # Patch embedding / Unpatchify
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size, image_size)
        self.unpatchify = Unpatchify(embed_dim, self.out_channels, patch_size, image_size)

        # Conditional embedding
        self.time_emb = SinusoidalPosEmb(timestep_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.class_emb = nn.Embedding(num_classes, embed_dim) if num_classes is not None else None

        # Transformer trunk
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, gate_scale=0.1)
            for _ in range(depth)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        # output head
        self.noise_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, timesteps, labels=None):
        B = x.shape[0]
        tokens, spatial = self.patch_embed(x)  # [B, N, D]

        # conditional embedding
        t_emb = self.time_mlp(self.time_emb(timesteps))
        cond = t_emb
        if self.class_emb is not None and labels is not None:
            cond = cond + self.class_emb(labels)

        # Transformer trunk
        h = tokens
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.final_ln(h)

        # output
        tokens_out = self.noise_head(h)
        pred_noise = self.unpatchify(tokens_out, spatial)
        return pred_noise


# -----------------------
# Example usage / Dummy run
# -----------------------
if __name__ == "__main__":
    B = 2
    C = 4
    H = W = 32
    model = DiT(image_size=H, in_channels=C, embed_dim=256, patch_size=1, depth=6, num_heads=8, num_classes=10, out_channels=C)
    latents = torch.randn(B, C, H, W)
    timesteps = torch.randint(0, 1000, (B,))
    labels = torch.randint(0, 10, (B,))
    pred_noise = model(latents, timesteps, labels)
    print("pred_noise", pred_noise.shape)  # [B, C, H, W]
