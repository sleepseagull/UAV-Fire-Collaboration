import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module (ECCV 2018)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    CBAM = Channel Attention + Spatial Attention（串联）
      1) Channel Attention: 生成 (B,C,1,1) 的通道权重
      2) Spatial Attention: 生成 (B,1,H,W) 的空间权重
      out = x * Mc * Ms（逐元素相乘，带广播）
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: 输入通道数 C（由 parse_model 注入，必须放第一个）
            reduction: 通道注意力 MLP 压缩比（常用 16）
            kernel_size: 空间注意力卷积核大小（论文常用 7）
        """
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)

        # -------------------------
        # 1) Channel Attention
        # -------------------------
        # GAP/GMP: (B,C,H,W)->(B,C,1,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP（用 1x1 Conv 等价实现）: C -> hidden -> C
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),  # (B,hidden,1,1)
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),  # (B,C,1,1)
        )

        # -------------------------
        # 2) Spatial Attention
        # -------------------------
        # 通道池化后 concat: (B,2,H,W) -> 7x7 conv -> (B,1,H,W)
        pad = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # =========================================================
        # 1) Channel Attention
        # =========================================================
        # avg: (B,C,1,1), mx: (B,C,1,1)
        avg = self.avg_pool(x)
        mx = self.max_pool(x)

        # 共享 MLP: (B,C,1,1)->(B,C,1,1)
        # Mc: (B,C,1,1)
        Mc = self.sigmoid(self.mlp(avg) + self.mlp(mx))

        # 通道加权（广播到 H,W）
        x_c = x * Mc  # (B,C,H,W)

        # =========================================================
        # 2) Spatial Attention
        # =========================================================
        # 通道维做 avg/max： (B,C,H,W)->(B,1,H,W)
        avg_c = x_c.mean(dim=1, keepdim=True)            # (B,1,H,W)
        max_c = x_c.max(dim=1, keepdim=True)[0]          # (B,1,H,W)

        # concat: (B,2,H,W)
        s = torch.cat([avg_c, max_c], dim=1)

        # Ms: (B,1,H,W)
        Ms = self.sigmoid(self.spatial_conv(s))

        # 空间加权（广播到通道维）
        out = x_c * Ms  # (B,C,H,W)

        return out
