import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible_groups(channels: int, groups: int) -> int:
    """让 GN 的 groups 一定能整除 channels（工程里更稳）"""
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ELA(nn.Module):
    """
    ELA: Efficient Local Attention (arXiv 2024)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    流程（与 CA 类似，但更轻更直接）：
      1) strip pooling: 得到 (B,C,H,1) 和 (B,C,1,W)
      2) 1D conv(沿 H/W) + GN: 生成两个方向的注意力权重
      3) sigmoid 后逐元素缩放：out = x * a_h * a_w
    """

    def __init__(self, channels: int, k: int = 7, gn_groups: int = 8, use_residual: bool = False):
        super().__init__()
        assert k % 2 == 1, "k 建议用奇数"

        self.C = channels
        self.k = k
        self.use_residual = use_residual

        g = _make_divisible_groups(channels, gn_groups)

        # 对输入做 GN（论文强调 GN 相比 BN 更稳）
        self.gn = nn.GroupNorm(num_groups=g, num_channels=channels)

        # 两个方向各一条 1D conv：保持通道数不变（不降维）
        # Conv1d 输入是 (B, C, L)
        self.conv_h = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=1, bias=True)
        self.conv_w = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        x_gn = self.gn(x)  # (B, C, H, W)

        # 1) strip pooling（沿某一维做平均）
        x_h = x_gn.mean(dim=3, keepdim=True)  # (B, C, H, 1)  在 W 上池化
        x_w = x_gn.mean(dim=2, keepdim=True)  # (B, C, 1, W)  在 H 上池化

        # 2) 变成 Conv1d 需要的 (B, C, L)
        x_h_1d = x_h.squeeze(3)              # (B, C, H)
        x_w_1d = x_w.squeeze(2)              # (B, C, W)

        # 3) 1D conv + sigmoid 得到方向权重
        a_h = self.sigmoid(self.conv_h(x_h_1d))  # (B, C, H)
        a_w = self.sigmoid(self.conv_w(x_w_1d))  # (B, C, W)

        # 4) reshape 回来用于广播相乘
        a_h = a_h.unsqueeze(3)  # (B, C, H, 1)
        a_w = a_w.unsqueeze(2)  # (B, C, 1, W)

        out = x * a_h * a_w     # (B, C, H, W)  广播到对应维度
        if self.use_residual:
            out = out + x
        return out
