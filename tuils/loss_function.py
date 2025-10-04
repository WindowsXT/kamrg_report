import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pandas as pd




class BinaryEntropyLoss_weight_pos_1(nn.Module):
    def __init__(
        self,
        df_labels: pd.DataFrame,
        size_average: bool = True,
        eps: float = 1e-6,
        device: torch.device = torch.device('cpu')
    ):
        """
        df_labels: 一个 DataFrame，每列都是 0/1 标签，
                   行数 = 样本数，列数 = 类别数
        size_average: 如果为 True，按 batch 大小取均值，否则 sum
        eps: 为了避免除零的小常数
        device: 计算用的设备
        """
        super().__init__()
        self.size_average = size_average

        # 1) 统计正负样本数
        pos_counts = df_labels.sum(axis=0).values.astype(float)  # shape (C,)
        total      = len(df_labels)
        neg_counts = total - pos_counts

        # 2) 负样本数 / 正样本数
        pos_weights = neg_counts / (pos_counts + eps)           # shape (C,)

        # 3) 注册到 buffer，并放到指定 device
        pw = torch.from_numpy(pos_weights).float().to(device)
        self.register_buffer('pos_weight', pw)  # 在 forward 时自动和 module 同 device

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)  —— 网络原始输出
        targets: (B, C) —— 0/1 标签
        """
        # binary_cross_entropy_with_logits 内部会先做 sigmoid
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction='mean' if self.size_average else 'sum'
        )
        return loss




class BinaryEntropyLoss_weight_pos_2(nn.Module):
    def __init__(self, df_labels: pd.DataFrame, size_average=True, device=torch.device('cpu')):
        super().__init__()
        self.size_average = size_average

        # df_labels 中每列是一个标签（14 列），值为 0/1
        pos_counts = df_labels.sum(axis=0).values  # shape (14,)
        total      = len(df_labels)
        # 计算 (1 - freq)^2
        weights    = (1 - pos_counts/total)**2
        
        # 将权重转为 Tensor，并保存到 Buffer
        self.register_buffer('class_weight', torch.FloatTensor(weights).to(device))

    def forward(self, input, target):
        # input: (batch,14)，target: (batch,14)
        # 扩成 (batch,14)
        w = self.class_weight.unsqueeze(0).expand_as(input)
        loss = F.binary_cross_entropy(input, target, weight=w, reduction='mean' if self.size_average else 'sum')
        return loss



