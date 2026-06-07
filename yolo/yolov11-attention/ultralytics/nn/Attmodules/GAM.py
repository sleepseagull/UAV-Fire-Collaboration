import torch
import torch.nn as nn


class GAM(nn.Module):
    """
    GAM: Global Attention Mechanism (2021)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    适配说明：
      - 你的 parse_model 会注入 channels=C 到 __init__ 的第一个参数，
        所以这里必须 __init__(channels, ...)

    典型 GAM 流程（顺序：Channel -> Spatial）：
      1) Channel Attention（跨维交互）：把空间展平，用 MLP 在通道维上建模全局依赖
      2) Spatial Attention：用卷积在空间上建模位置相关性
      out = x * Mc * Ms  （逐元素相乘，广播）
    """

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        """
        Args:
            channels: 输入通道数 C（parse_model 注入）
            reduction: 通道分支 MLP 压缩比（GAM 常用 4）
            spatial_kernel: 空间分支卷积核大小（常用 7）
        """
        super().__init__()
        assert channels > 0
        self.channels = channels
        hidden = max(1, channels // reduction)

        # -------------------------
        # 1) Channel Attention（MLP 在通道维做全局交互）
        # -------------------------
        # 思路：先把 (H,W) 展平为 N=H*W，把每个位置的 C 维向量送进 MLP，
        # MLP 输出仍是 C 维，最后再 reshape 回 (B,C,H,W) 得到 Mc（同形状权重）
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        # -------------------------
        # 2) Spatial Attention（卷积建模空间相关）
        # -------------------------
        pad = spatial_kernel // 2
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels // reduction if channels // reduction > 0 else 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction if channels // reduction > 0 else 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction if channels // reduction > 0 else 1, channels, kernel_size=spatial_kernel, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # =========================================================
        # 1) Channel Attention
        # =========================================================
        # 先把 x 变形为 (B, N, C)
        # x_n: (B, N, C)
        x_n = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        # 逐位置 MLP： (B,N,C)->(B,N,C)
        # 注意：Linear 会作用在最后一维 C 上
        mc = self.channel_mlp(x_n)

        # mc: (B,N,C) -> (B,C,H,W)
        mc = self.channel_sigmoid(mc).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 通道注意力加权
        x_c = x * mc  # (B,C,H,W)

        # =========================================================
        # 2) Spatial Attention
        # =========================================================
        # ms: (B,C,H,W)
        ms = self.spatial_sigmoid(self.spatial(x_c))

        # 空间注意力加权（与 x_c 同形状，因此逐元素乘）
        out = x_c * ms  # (B,C,H,W)

        return out
