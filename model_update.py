### IMPORTS ###
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

### SET GPU OR CPU ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU device detected: {device}")
else:
    device = torch.device("cpu")
    print(f"GPU device not detected. Using CPU: {device}")


### DIFFUSION ###
class Diffusion():
    def __init__(self, device=device, timesteps=1000, beta_start=0.0001, beta_end=0.02):
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

    

### MODEL ###
def get_time_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

class DDPM_Predict(nn.Module):
    def __init__(self, ESM_emb=1281, emb_dim=1024, num_layers=4, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Embedding layers
        self.seq_embedding = nn.Linear(ESM_emb, emb_dim).to(device)
        self.bepi_project = nn.Linear(1, emb_dim).to(device)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True
        ).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        
        # Output projection
        self.output_proj = nn.Linear(emb_dim, 1).to(device)

    def forward(self, noisy_mask, seq, t, attention_mask=None):
        """
        noisy_mask: (B, L, 1)
        seq:        (B, L, ESM_emb)
        t:          (B,)
        attention_mask: (B, L) - torch.bool, True 表示有效，False 为 padding
        """
        B, L, D = seq.shape
        attention_mask = attention_mask.to(device)
        seq_emb = self.seq_embedding(seq.to(device))             # (B, L, D)
        noisy_mask_emb = self.bepi_project(noisy_mask.to(device))  # (B, L, D)
        time_emb = get_time_embedding(t.to(device).unsqueeze(-1), self.emb_dim)  # (B, D)
        time_emb = time_emb.unsqueeze(1).expand(B, L, -1)  # (B, L, D)
        
        x = seq_emb + noisy_mask_emb + time_emb  # (B, L, D)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # Transformer中True表示有效
        else:
            key_padding_mask = None
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L, D)

        out = self.output_proj(x).squeeze(-1)  # (B, L)
        
        return out

class DDPM_enhance(nn.Module):
    def __init__(self, ESM_emb=1281, emb_dim=1024, num_layers=4, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim

        # Embedding layers
        self.seq_embedding = nn.Linear(ESM_emb, emb_dim).to(device)
        self.bepi_project = nn.Linear(1, emb_dim).to(device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        # Output projection
        self.output_proj = nn.Linear(emb_dim, 1)  # Predict noise or mask value


    def forward(self, noisy_mask, seq, bepi_scores, t, attention_mask=None):
        """
        noisy_mask: (B, L, 1) - optional, for conditioning or residual use
        seq:        (B, L, ESM_emb)
        bepi_scores:(B, L) - float [0~1]
        t:          (B,) - time step
        attention_mask: (B, L) - torch.bool, True 表示有效，False 为 padding
        """
        attention_mask = attention_mask.to(device)
        B, L, D = seq.shape
        # assert L <= self.seq_len

        # Embedding inputs
        seq_emb = self.seq_embedding(seq.to(device))                              # (B, L, emb_dim)
        noisy_mask_emb = self.bepi_project(noisy_mask.to(device))                 # (B, L, emb_dim)
        bepi_emb = self.bepi_project(bepi_scores.unsqueeze(-1).to(device))        # (B, L, emb_dim)
        time_emb = get_time_embedding(t.unsqueeze(-1), self.emb_dim)   # (B, emb_dim)
        time_emb = time_emb.unsqueeze(1).expand(B, L, -1)              # (B, L, emb_dim)
        
        # Combine all embeddings
        x = seq_emb + bepi_emb + noisy_mask_emb + time_emb                             # (B, L, emb_dim)

        # Optional: add residual noisy mask as conditioning
        # x = x + noisy_mask  # If you want to directly feed noisy mask values
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # Transformer中True表示有效
        else:
            key_padding_mask = None
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)             # (B, L, emb_dim)  

        # Project to 1D mask prediction
        out = self.output_proj(x).squeeze(-1)                          # (B, L)

        return out