import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit_block import DiTBlock


# -----------------------
# Utilities / Embeddings
# -----------------------
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding (like in DDPM / Transformer positional enc)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: tensor shape [B] (scalars, timesteps)
        returns: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        emb = t[:, None].float() * emb[None, :]  # [B, half]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]


# -----------------------
# Patchify / Unpatchify
# -----------------------
# class PatchEmbed(nn.Module):
#     """
#     Patchify and linear embedding.
#     Input: latents of shape [B, C, H, W]
#     Splits into patches of size patch_size x patch_size, flatten each patch and linear-project to embed_dim.
#     Output tokens: [B, N, embed_dim], where N = (H//p)*(W//p)
#     """
#     def __init__(self, in_channels, embed_dim, patch_size):
#         super().__init__()
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         # We'll implement patchify as a conv with stride=patch_size, kernel=patch_size
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         # x: [B, C, H, W]
#         x = self.proj(x)  # [B, embed_dim, H/p, W/p]
#         B, D, Hs, Ws = x.shape
#         x = x.flatten(2).transpose(1, 2)  # -> [B, N, D] where N = Hs*Ws
#         return x, (Hs, Ws)


# class Unpatchify(nn.Module):
#     """
#     Convert tokens [B, N, embed_dim] back to [B, C, H, W] roughly by ConvTranspose
#     We'll map embed_dim -> out_channels with ConvTranspose2d
#     """
#     def __init__(self, embed_dim, out_channels, patch_size, out_hw=None):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.out_channels = out_channels
#         self.patch_size = patch_size
#         # We'll use a simple conv transpose that reverses PatchEmbed
#         self.deproj = nn.ConvTranspose2d(embed_dim, out_channels,
#                                          kernel_size=patch_size,
#                                          stride=patch_size)

#     def forward(self, tokens, spatial_shape):
#         # tokens: [B, N, D], spatial_shape: (Hs, Ws) where Hs = H/patch
#         B, N, D = tokens.shape
#         Hs, Ws = spatial_shape
#         x = tokens.transpose(1, 2).reshape(B, D, Hs, Ws)  # [B, D, Hs, Ws]
#         x = self.deproj(x)  # [B, out_channels, H, W]
#         return x

    
# -----------------------
# DiT (最小可训练版本)
# -----------------------
class DiT(nn.Module):
    def __init__(self,
                 image_size=28,
                 in_channels=1,
                 patch_size=4,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 timestep_emb_dim=256,
                 num_classes=None,   # 如果有类条件
                 out_channels=None   # 默认 = in_channels
                 ):
        super().__init__()
        self.image_size  = image_size
        self.in_channels = in_channels
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.num_classes = num_classes
        self.out_channels = out_channels or in_channels

        assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"
        self.h_patches = image_size // patch_size
        self.w_patches = image_size // patch_size
        self.num_patches = self.h_patches * self.w_patches

        # 每个 token 的原始维度（像素展平）
        self.patch_dim_in  = in_channels  * patch_size * patch_size
        self.patch_dim_out = self.out_channels * patch_size * patch_size

        # token 投影
        self.token_proj = nn.Linear(self.patch_dim_in, embed_dim)

        # 条件嵌入 (time + optional class)
        self.time_emb = SinusoidalPosEmb(timestep_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_emb = nn.Embedding(num_classes, embed_dim) if num_classes is not None else None

        # Transformer trunk
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, gate_scale=0.1)
            for _ in range(depth)
        ])
        self.final_ln  = nn.LayerNorm(embed_dim)

        # 输出 head：把 token 从 embed_dim 映回 patch 像素
        self.noise_head = nn.Linear(embed_dim, self.patch_dim_out)
        nn.init.zeros_(self.noise_head.bias)

    # ===== Patchify (保留通道) =====
    def patchify(self, x):
        """
        x: [B, C, H, W]
        return: tokens [B, N, C*p*p]
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h = H // p
        w = W // p
        # [B, C, h, p, w, p] -> [B, h, w, C, p, p] -> [B, N, C*p*p]
        x = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        tokens = x.reshape(B, h * w, C * p * p)
        return tokens

    # ===== Unpatchify (还原通道和空间) =====
    def unpatchify(self, tokens):
        """
        tokens: [B, N, C*p*p] (这里的 C 是 out_channels)
        return: [B, C, H, W]
        """
        B, N, D = tokens.shape
        p = self.patch_size
        h = self.h_patches
        w = self.w_patches
        C = self.out_channels
        assert N == h * w and D == C * p * p, f"tokens 形状不匹配{N}, {h}, {w}, {C}, {p}"
        x = tokens.reshape(B, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, h * p, w * p)
        return x

    def forward(self, latents, timesteps, labels=None):
        """
        latents:   [B, C_in, H, W] (加噪图像/潜变量)
        timesteps: [B] (long/int)
        labels:    [B] (optional class)
        return:
          pred_noise: [B, C_out, H, W]
        """
        # 1) patchify → token_proj
        tokens = self.patchify(latents)                    # [B, N, C_in*p*p]
        x = self.token_proj(tokens)                        # [B, N, D]

        # 2) 条件嵌入
        t_emb = self.time_mlp(self.time_emb(timesteps))    # [B, D]
        cond = t_emb
        if (self.class_emb is not None) and (labels is not None):
            cond = cond + self.class_emb(labels)           # [B, D]

        # 3) Transformer trunk
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.final_ln(x)

        # 4) 输出 head → unpatchify
        noise_tokens = self.noise_head(x)                  # [B, N, C_out*p*p]
        pred_noise   = self.unpatchify(noise_tokens)       # [B, C_out, H, W]
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
