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
class PatchEmbed(nn.Module):
    """
    Patchify and linear embedding.
    Input: latents of shape [B, C, H, W]
    Splits into patches of size patch_size x patch_size, flatten each patch and linear-project to embed_dim.
    Output tokens: [B, N, embed_dim], where N = (H//p)*(W//p)
    """
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # We'll implement patchify as a conv with stride=patch_size, kernel=patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        B, D, Hs, Ws = x.shape
        x = x.flatten(2).transpose(1, 2)  # -> [B, N, D] where N = Hs*Ws
        return x, (Hs, Ws)


class Unpatchify(nn.Module):
    """
    Convert tokens [B, N, embed_dim] back to [B, C, H, W] roughly by ConvTranspose
    We'll map embed_dim -> out_channels with ConvTranspose2d
    """
    def __init__(self, embed_dim, out_channels, patch_size, out_hw=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        # We'll use a simple conv transpose that reverses PatchEmbed
        self.deproj = nn.ConvTranspose2d(embed_dim, out_channels,
                                         kernel_size=patch_size,
                                         stride=patch_size)

    def forward(self, tokens, spatial_shape):
        # tokens: [B, N, D], spatial_shape: (Hs, Ws) where Hs = H/patch
        B, N, D = tokens.shape
        Hs, Ws = spatial_shape
        x = tokens.transpose(1, 2).reshape(B, D, Hs, Ws)  # [B, D, Hs, Ws]
        x = self.deproj(x)  # [B, out_channels, H, W]
        return x

# -----------------------
# Full DiT Model
# -----------------------
class DiT(nn.Module):
    def __init__(self,
                 in_channels=4,       # latent channels (e.g., 4)
                 embed_dim=512,       # transformer hidden dim
                 patch_size=1,
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.0,
                 timestep_emb_dim=512,
                 num_classes=None,    # if conditional on class labels
                 out_channels=4       # predict noise channels (same as in_channels)
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads

        # Patchify
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.unpatchify = Unpatchify(embed_dim, out_channels, patch_size)

        # optional class embedding
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, embed_dim)
        else:
            self.class_emb = None

        # timestep embedding (sinusoidal + MLP to project to embed_dim)
        self.time_emb = SinusoidalPosEmb(timestep_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # trunk: stack of DiTBlocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # final norm and head
        self.final_ln = nn.LayerNorm(embed_dim)
        # Two heads: predict noise and predict sigma (optional)
        # 修复：将输出维度改为embed_dim，与Unpatchify的输入要求匹配
        self.noise_head = nn.Linear(embed_dim, embed_dim)
        self.sigma_head = nn.Linear(embed_dim, embed_dim)

        # init heads small
        nn.init.zeros_(self.noise_head.bias)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, latents, timesteps, labels=None):
        """
        latents: [B, C, H, W] (noised latents)
        timesteps: [B] (long tensor)
        labels: [B] (optional)
        returns:
          pred_noise: [B, C, H, W]
          pred_sigma: [B, C, H, W]
        """
        B = latents.shape[0]
        tokens, spatial = self.patch_embed(latents)  # [B, N, D], (Hs, Ws)
        # Conditioning embedding
        t_emb = self.time_emb(timesteps)  # [B, t_dim]
        t_emb = self.time_mlp(t_emb)      # [B, embed_dim]

        if self.class_emb is not None and labels is not None:
            c_emb = self.class_emb(labels)  # [B, embed_dim]
            cond = t_emb + c_emb
        else:
            cond = t_emb  # [B, embed_dim]

        # Pass through DiT blocks
        x = tokens  # [B, N, D]
        for blk in self.blocks:
            x = blk(x, cond)  # cond broadcast inside block to per-token shift/scale

        # final transform
        x = self.final_ln(x)  # [B, N, D]

        # heads: produce per-token outputs which we unpatchify
        # 修复：直接输出embed_dim维度的token
        noise_tokens = self.noise_head(x)  # [B, N, embed_dim]
        sigma_tokens = self.sigma_head(x)  # [B, N, embed_dim]

        # 简化：直接传递给unpatchify，无需复杂reshape
        pred_noise = self.unpatchify(noise_tokens, spatial)  # [B, out_channels, H, W]
        pred_sigma = self.unpatchify(sigma_tokens, spatial)

        return pred_noise


# -----------------------
# Example usage / Dummy run
# -----------------------
if __name__ == "__main__":
    B = 2
    C = 4
    H = W = 32
    model = DiT(in_channels=C, embed_dim=256, patch_size=1, depth=6, num_heads=8, num_classes=10, out_channels=C)
    latents = torch.randn(B, C, H, W)
    timesteps = torch.randint(0, 1000, (B,))
    labels = torch.randint(0, 10, (B,))
    pred_noise = model(latents, timesteps, labels)
    print("pred_noise", pred_noise.shape)  # [B, C, H, W]
