import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module (ICML 2021)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    说明：
      - 为了适配 Ultralytics v11 的 parse_model（会自动注入 channels=C），
        这里 __init__ 的第一个参数必须接收 channels（即使 SimAM 本身不依赖 C）。
      - 注意力权重为逐元素 (B,C,H,W)，属于“像素/位置级”重标定。
    """

    def __init__(self, channels: int, e_lambda: float = 1e-4):
        """
        Args:
            channels: 输入通道数 C（由 parse_model 自动注入；SimAM 内部不使用，仅占位适配）
            e_lambda: 稳定项（避免除零），论文/常用实现一般取 1e-4
        """
        super().__init__()
        self.channels = channels  # 可选：存一下方便 debug（不影响计算）
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # ---- Step A: 通道均值 ----
        # mu: (B, C, 1, 1)
        mu = x.mean(dim=(2, 3), keepdim=True)

        # ---- Step B: 偏差平方 ----
        # d: (B, C, H, W)
        d = (x - mu).pow(2)

        # ---- Step C: 通道内均方（方差项）----
        # v: (B, C, 1, 1)
        v = d.mean(dim=(2, 3), keepdim=True)

        # ---- Step D: 能量评分（常用简化实现）----
        # score: (B, C, H, W)
        score = d / (4.0 * (v + self.e_lambda)) + 0.5

        # ---- Step E: 注意力权重（逐元素）----
        # attn: (B, C, H, W)
        attn = self.sigmoid(score)

        # ---- Step F: 重标定输出 ----
        # out: (B, C, H, W)
        return x * attn
