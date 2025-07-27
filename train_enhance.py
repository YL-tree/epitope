### Import ###
from model_update import DDPM_enhance, Diffusion
from data import BepiPredDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm
import random
from pathlib import Path
import matplotlib.pyplot as plt

random.seed(418)  # 设置种子值

### SET GPU OR CPU ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU device detected: {device}")
else:
    device = torch.device("cpu")
    print(f"GPU device not detected. Using CPU: {device}")


def map_back_to_unit_range(y, mu=0.0, sigma=1.0):
    return 0.5 * (1 + torch.erf((y - mu) / (sigma * torch.sqrt(torch.tensor(2.)))))

def collate_fn(batch):
    """
    输入: batch 是一个列表，包含若干个 (acc, esm_embedding, epitope_label)
    输出: padded 的 batch，包括 mask
    """
    accs, embeddings, labels, bepi_pred = zip(*batch)

    # pad_sequence 默认按第一维对齐，batch_first=True 输出形状为 (B, L, D)
    padded_embeddings = pad_sequence(embeddings, batch_first=True)  # (B, L, D)
    padded_labels = pad_sequence(labels, batch_first=True)          # (B, L)
    padded_bepi_pred = pad_sequence(bepi_pred, batch_first=True)    # (B, L)


    # 创建 attention mask：True 表示有效位置，False 表示padding
    attention_mask = torch.zeros_like(padded_labels, dtype=torch.bool)
    for i, label in enumerate(labels):
        attention_mask[i, :label.size(0)] = True

    return accs, padded_embeddings, padded_labels, padded_bepi_pred, attention_mask
    

### checkpoint ###
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

### train ###
def train(model, dataloader, diffusion, optimizer, steps=1000, device=device, epochs=3, checkpoint_dir="./checkpoints", model_path="full_model.pth"):
    model.train()
    epochs_losses = []  # 记录每个epoch的平均损失

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        loss_record= []
        for acc, esm, mask, bepi_pred, attention_mask in progress_bar:  # esm: 真实ESM嵌入 [B, L, D]
            esm = esm.to(device)
            mask = mask.to(device)
            bepi_pred = bepi_pred.to(device)
            attention_mask = attention_mask.to(device)

            # 1. 随机采样时间步和噪声
            t = torch.randint(0, steps, (mask.size(0),), device=device)
            noise = torch.randn_like(mask).to(device)
            
            # 2. 前向加噪（根据噪声调度）
            noisy_mask = diffusion.q_sample(mask.unsqueeze(-1), t)
            t = t.float()

            # 3. 预测噪声并计算损失
            pred_noise = model(noisy_mask, esm, bepi_pred, t, attention_mask)

            # 4. 计算表位分类损失
            loss = F.mse_loss(pred_noise, noise)
            
            # 4. 反向传播
            optimizer.zero_grad()
            loss.backward()
            loss_record.append(loss.item())
            optimizer.step()
            
            # 5. 记录损失
            progress_bar.set_postfix({"loss": loss.item()})
            epoch_loss = torch.tensor(loss_record).mean().item()

        epochs_losses.append(epoch_loss)
        # 每15个epoch保存一次checkpoint
        if (epoch + 1) % 15 == 0:
            save_checkpoint(epoch + 1, model, optimizer, epoch_loss, checkpoint_dir)
            print(f"Epoch {epoch + 1}, Loss_Mean: {epoch_loss}", end="\r")
            
    # 保存整个模型和最后的损失图
    torch.save(model, model_path)
    plt.plot(epochs_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')
    print(f"Full model saved to {model_path}")

    

@torch.no_grad()
def ddpm_sampling(model, diffusion, esm_seq, bepi_scores, attention_mask, num_steps=10):
    """
    Args:
        model: 训练好的 EpitopeTransformerDDPM 模型
        diffusion: 噪声调度器（如 SimpleNoiseScheduler）
        esm_seq: (B, L) 蛋白质序列
        bepi_scores: (B, L) BepiPred得分（0~1）
        attention_mask: (B, L) 注意力掩码
        num_steps: T，采样步数
        
    Returns:
        x0: (B, L) 最终生成的掩码（0~1）
    """

    B, L,  _= esm_seq.shape
    x_t = torch.randn(B, L, 1).to(device)  # 初始纯噪声

    for t_index in range(num_steps - 1, -1, -1):
        t = torch.full((B,), t_index, device=device, dtype=torch.long)
        # 模型预测噪声
        pred_noise = model(x_t, esm_seq, bepi_scores, t, attention_mask).unsqueeze(-1)  # (B, L, 1)

        x_t = diffusion.p_sample(x_t, pred_noise, t, t_index)
        
    # 输出最终 x0
    x0 = x_t.squeeze(-1)  # (B, L)

    return x0  # 每个位点为表位的概率分布

# 保存采样结果为 .pt 文件
def save_tensor(tensor, path):
    torch.save(tensor, path)
    print(f"Tensor saved to {path}")
    
def sample(model, diffusion, test_dataloader):
    model.eval()
    with torch.no_grad():
        for acc, esm, mask, bepi_pred, attention_mask in tqdm(test_dataloader, desc="Sampling Progress"):
            esm = esm.to(device)
            mask = mask.to(device)
            bepi_pred = bepi_pred.to(device)
            attention_mask = attention_mask.to(device)

            pred_mask = ddpm_sampling(model, diffusion, esm, bepi_pred, attention_mask)

            # 逐个样本处理
            for idx in range(len(mask)):
                valid_indices = attention_mask[idx].bool()
                clean = mask[idx][valid_indices]
                print(bepi_pred.shape, mask.shape)
                b_pred = bepi_pred[idx][valid_indices]
                pred = pred_mask[idx][valid_indices]
                if not os.path.exists(f"data/predict_enhance"):
                    os.makedirs(f"data/predict_enhance")
                save_path = f"data/predict_enhance/{acc[idx]}.pt"
                save_tensor({
                    "acc": acc[idx],
                    "bepi_pred": b_pred,
                    "mask": map_back_to_unit_range(clean),
                    "pred_mask": map_back_to_unit_range(pred),
                }, save_path)



if __name__ == "__main__":
    # 获取所有ESM编码文件路径列表
    bepi_files = list(Path("data/BepiPred_outputs").glob("*.pt"))

    # 划分训练集和测试集
    train_files, test_files = train_test_split(bepi_files, test_size=0.2, random_state=42)
    train_bepi_dataset = BepiPredDataset(train_files)
    test_bepi_dataset = BepiPredDataset(test_files)

    train_bepi_dataloader = DataLoader(train_bepi_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_bepi_dataloader = DataLoader(test_bepi_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    checkpoint_dir = './checkpoints_enhance'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    DDPM_Enhance_model = DDPM_enhance()
    diffusion = Diffusion()
    optimizer = optim.Adam(DDPM_Enhance_model.parameters(), lr=0.0001)

    train(DDPM_Enhance_model, train_bepi_dataloader, diffusion, optimizer)
    
    sample(DDPM_Enhance_model, diffusion, test_bepi_dataloader)
