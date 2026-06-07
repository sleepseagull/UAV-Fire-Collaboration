import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    ECA (Efficient Channel Attention)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    核心思想：
      - 先 GAP 得到通道描述子 (B, C, 1, 1)
      - 不做 SE 的两层 MLP，而是对通道维做 1D 卷积建模局部通道交互
      - Sigmoid 得到通道权重 (B, C, 1, 1)，再与输入逐通道相乘
    """

    def __init__(self, channels: int, k_size: int = None, gamma: int = 2, b: int = 1):
        """
        Args:
            channels: 输入通道数 C
            k_size: 1D 卷积核大小（奇数）。若为 None，则按论文自适应计算
            gamma, b: 自适应核大小的超参数（论文给的默认值常用 gamma=2, b=1）
        """
        super().__init__()
        assert channels > 0

        # 1) GAP: (B,C,H,W) -> (B,C,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 2) 自适应选择卷积核大小（保证为奇数）
        if k_size is None:
            # k = |(log2(C)/gamma) + b|_odd
            t = int(abs((math.log2(channels) / gamma) + b))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)  # 实务中常设最小 3

        # 3) 在通道维做 1D conv：把 (B,C,1,1) 变成 (B,1,C) 再卷积
        # Conv1d 输入: (B, 1, C) -> 输出: (B, 1, C)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ---- Squeeze ----
        # y: (B, C, 1, 1)
        y = self.gap(x)

        # ---- Channel interaction (1D conv on channel dim) ----
        # (B, C, 1, 1) -> squeeze -> (B, C)
        # (B, C) -> unsqueeze to (B, 1, C) for Conv1d
        y = y.squeeze(-1).squeeze(-1)        # (B, C)
        y = y.unsqueeze(1)                   # (B, 1, C)
        y = self.conv1d(y)                   # (B, 1, C)

        # ---- Gate ----
        # (B, 1, C) -> (B, C, 1, 1)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # ---- Scale ----
        # 广播乘法: (B,C,H,W) * (B,C,1,1) -> (B,C,H,W)
        return x * y
