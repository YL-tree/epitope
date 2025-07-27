class Config:
    # 训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    epochs = 500 
    batch_size = 32
    lr = 1e-4 

    # 模型参数
    window_size = 20

    # 设置断点
    checkpoint_interval = 25

    # transformer部分参数
    output_dim = window_size  # 输出词汇表大小
    d_model = 512  # 词嵌入维度
    num_head = 8  # 多头注意力头数
    num_encoder_layers = 3 # 编码器层数
    num_decoder_layers = 3 # 解码器层数 
    dim_feedforward = 2048  # 前馈神经网络隐藏层维度
    dropout = 0.1  # Dropout 概率

    # diffusion部分参数
    timesteps = 1000  # 扩散步数
    beta_start = 0.0001  # Beta 起始值
    beta_end = 0.02  # Beta 结束值

config = Config()
